#pragma once

#include "engine/engine.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward decls from llama.cpp C API
struct llama_model;
struct llama_context;
using llama_token = int32_t;

namespace minisgl::engine {

struct LlamaCppOptions {
  std::string model_path;
  int n_ctx = 4096;
  int n_threads = 0;     // 0 -> llama.cpp default
  int n_gpu_layers = -1; // -1 -> llama.cpp default
  int seed = -1;
};

class LlamaCppEngine final : public Engine {
public:
  explicit LlamaCppEngine(LlamaCppOptions opts);
  ~LlamaCppEngine() override;

  GenerateResult Generate(const std::string &model,
                          const std::string &prompt,
                          const SamplingParams &params,
                          const std::string &session_id) override;

private:
  struct Session {
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx;
    std::vector<llama_token> tokens;
  };

  Session &get_or_create_session_(const std::string &session_id);
  std::vector<llama_token> tokenize_(const std::string &text, bool add_bos);
  void eval_(llama_context *ctx, const std::vector<llama_token> &tokens);
  llama_token sample_greedy_(llama_context *ctx) const;
  std::string token_to_piece_(llama_context *ctx, llama_token tok) const;

private:
  LlamaCppOptions opts_;
  std::unique_ptr<llama_model, void (*)(llama_model *)> model_;

  mutable std::mutex mu_;
  std::unordered_map<std::string, Session> sessions_;
};

} // namespace minisgl::engine


