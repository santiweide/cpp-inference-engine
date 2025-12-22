#include "engine/llama_cpp_engine.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

// llama.cpp headers (expect include path to point to llama.cpp/include)
#include <llama.h>

namespace minisgl::engine {
namespace {

inline auto now_ms() -> int64_t {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

} // namespace

LlamaCppEngine::LlamaCppEngine(LlamaCppOptions opts)
    : opts_(std::move(opts)),
      model_(nullptr, [](llama_model *m) {
        if (m)
          llama_model_free(m);
      }) {
  if (opts_.model_path.empty()) {
    throw std::runtime_error("LlamaCppEngine: model_path is empty");
  }

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  if (opts_.n_gpu_layers != 0) {
    mparams.n_gpu_layers = opts_.n_gpu_layers;
  }

  llama_model *m = llama_model_load_from_file(opts_.model_path.c_str(), mparams);
  if (!m) {
    throw std::runtime_error("Failed to load llama.cpp model: " + opts_.model_path);
  }
  model_.reset(m);
}

LlamaCppEngine::~LlamaCppEngine() {
  // llama.cpp uses global backend state
  llama_backend_free();
}

LlamaCppEngine::Session &LlamaCppEngine::get_or_create_session_(const std::string &session_id) {
  const std::string key = session_id.empty() ? "__default__" : session_id;
  auto it = sessions_.find(key);
  if (it != sessions_.end()) {
    return it->second;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = opts_.n_ctx;
  if (opts_.seed != -1) {
    cparams.seed = opts_.seed;
  }
  if (opts_.n_threads > 0) {
    cparams.n_threads = opts_.n_threads;
  }

  llama_context *ctx_raw = llama_init_from_model(model_.get(), cparams);
  if (!ctx_raw) {
    throw std::runtime_error("Failed to create llama.cpp context");
  }

  Session s{std::unique_ptr<llama_context, void (*)(llama_context *)>(
                ctx_raw, [](llama_context *c) {
                  if (c)
                    llama_free(c);
                }),
            {}};

  auto [ins, _] = sessions_.emplace(key, std::move(s));
  return ins->second;
}

std::vector<llama_token> LlamaCppEngine::tokenize_(const std::string &text, bool add_bos) {
  std::vector<llama_token> out;
  out.resize(text.size() + 8);
  const int n =
      llama_tokenize(model_.get(), text.c_str(), static_cast<int>(text.size()), out.data(),
                     static_cast<int>(out.size()), add_bos, /*special=*/true);
  if (n < 0) {
    // need bigger buffer
    out.resize(static_cast<size_t>(-n));
    const int n2 =
        llama_tokenize(model_.get(), text.c_str(), static_cast<int>(text.size()), out.data(),
                       static_cast<int>(out.size()), add_bos, /*special=*/true);
    if (n2 < 0) {
      throw std::runtime_error("llama_tokenize failed");
    }
    out.resize(static_cast<size_t>(n2));
  } else {
    out.resize(static_cast<size_t>(n));
  }
  return out;
}

void LlamaCppEngine::eval_(llama_context *ctx, const std::vector<llama_token> &tokens) {
  if (tokens.empty())
    return;
  llama_batch batch = llama_batch_init(static_cast<int>(tokens.size()), /*embd=*/0, /*n_seq_max=*/1);
  for (int i = 0; i < (int)tokens.size(); ++i) {
    batch.token[i] = tokens[(size_t)i];
    batch.pos[i] = llama_get_kv_cache_used_cells(ctx) + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = false;
  }
  batch.logits[(int)tokens.size() - 1] = true;

  const int rc = llama_decode(ctx, batch);
  llama_batch_free(batch);
  if (rc != 0) {
    throw std::runtime_error("llama_decode failed");
  }
}

llama_token LlamaCppEngine::sample_greedy_(llama_context *ctx) const {
  const float *logits = llama_get_logits(ctx);
  const int n_vocab = llama_n_vocab(model_.get());
  int best = 0;
  float best_val = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best = i;
    }
  }
  return (llama_token)best;
}

std::string LlamaCppEngine::token_to_piece_(llama_context *ctx, llama_token tok) const {
  std::string s;
  s.resize(16);
  const int n = llama_token_to_piece(model_.get(), tok, s.data(), (int)s.size(), 0, true);
  if (n < 0) {
    s.resize((size_t)(-n));
    const int n2 = llama_token_to_piece(model_.get(), tok, s.data(), (int)s.size(), 0, true);
    if (n2 < 0) {
      return "";
    }
    s.resize((size_t)n2);
    return s;
  }
  s.resize((size_t)n);
  (void)ctx;
  return s;
}

GenerateResult LlamaCppEngine::Generate(const std::string &model,
                                       const std::string &prompt,
                                       const SamplingParams &params,
                                       const std::string &session_id) {
  (void)model; // server-level model selection is via opts_.model_path for now

  std::lock_guard<std::mutex> lk(mu_);
  auto &s = get_or_create_session_(session_id);
  llama_context *ctx = s.ctx.get();

  // Tokenize prompt. For new sessions, add BOS; otherwise we only append.
  const bool add_bos = s.tokens.empty();
  auto prompt_toks = tokenize_(prompt, add_bos);

  // Evaluate prompt tokens and update session history.
  eval_(ctx, prompt_toks);
  s.tokens.insert(s.tokens.end(), prompt_toks.begin(), prompt_toks.end());

  const llama_token eos = llama_token_eos(model_.get());

  GenerateResult result;
  result.prompt_tokens = (int)prompt_toks.size();

  std::string out_text;
  out_text.reserve(1024);

  const int max_new = params.max_tokens > 0 ? params.max_tokens : 128;
  for (int i = 0; i < max_new; ++i) {
    // Greedy sampling for now. (You can add temperature/top-k later.)
    const llama_token next = sample_greedy_(ctx);
    if (!params.ignore_eos && next == eos) {
      break;
    }
    // Append piece.
    out_text += token_to_piece_(ctx, next);

    // Evaluate next token.
    std::vector<llama_token> one{next};
    eval_(ctx, one);
    s.tokens.push_back(next);
    result.completion_tokens += 1;
  }

  result.text = out_text;
  return result;
}

} // namespace minisgl::engine


