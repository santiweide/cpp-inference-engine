#include "server.h"

#include "engine/dummy_engine.h"
#if __has_include("llama.h")
#include "engine/llama_cpp_engine.h"
#define MINISGL_HAS_LLAMA_CPP 1
#else
#define MINISGL_HAS_LLAMA_CPP 0
#endif

#include <chrono>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "minisgl_inference.grpc.pb.h"
#include "minisgl_inference.pb.h"

namespace minisgl {
namespace {

class InferenceServiceImpl final : public minisgl::inference::InferenceService::Service {
public:
  explicit InferenceServiceImpl(std::unique_ptr<minisgl::engine::Engine> engine)
      : engine_(std::move(engine)) {}

  grpc::Status Health(grpc::ServerContext *,
                      const minisgl::inference::HealthRequest *,
                      minisgl::inference::HealthResponse *resp) override {
    resp->set_ok(true);
    resp->set_message("ok");
    return grpc::Status::OK;
  }

  grpc::Status Generate(grpc::ServerContext *,
                        const minisgl::inference::GenerateRequest *req,
                        minisgl::inference::GenerateResponse *resp) override {
    const auto start = std::chrono::steady_clock::now();

    minisgl::engine::SamplingParams params;
    params.max_tokens = req->max_tokens() > 0 ? req->max_tokens() : 128;
    params.temperature = req->temperature();
    params.top_k = req->top_k() > 0 ? req->top_k() : 1;
    params.ignore_eos = req->ignore_eos();

    const auto r =
        engine_->Generate(req->model(), req->prompt(), params, req->session_id());

    resp->set_text(r.text);
    resp->set_prompt_tokens(r.prompt_tokens);
    resp->set_completion_tokens(r.completion_tokens);

    const auto end = std::chrono::steady_clock::now();
    resp->set_latency_ms(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    return grpc::Status::OK;
  }

  grpc::Status StreamGenerate(
      grpc::ServerContext *,
      const minisgl::inference::GenerateRequest *req,
      grpc::ServerWriter<minisgl::inference::GenerateChunk> *writer) override {
    // Default implementation: call Generate once and stream it as one chunk.
    minisgl::inference::GenerateResponse full;
    auto st = Generate(nullptr, req, &full);
    if (!st.ok()) {
      return st;
    }
    minisgl::inference::GenerateChunk chunk;
    chunk.set_token_text(full.text());
    chunk.set_finished(true);
    writer->Write(chunk);
    return grpc::Status::OK;
  }

private:
  std::unique_ptr<minisgl::engine::Engine> engine_;
};

} // namespace

struct Server::Impl {
  Options opts;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<InferenceServiceImpl> svc;
};

Server::Server(Options opts) : impl_(std::make_unique<Impl>()) {
  impl_->opts = std::move(opts);
}

Server::~Server() = default;

void Server::RunBlocking() {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(impl_->opts.listen_addr, grpc::InsecureServerCredentials());

  std::unique_ptr<minisgl::engine::Engine> engine;
  if (impl_->opts.backend == "dummy") {
    engine = std::make_unique<minisgl::engine::DummyEngine>();
  } else if (impl_->opts.backend == "llama_cpp") {
#if MINISGL_HAS_LLAMA_CPP
    minisgl::engine::LlamaCppOptions eopts;
    eopts.model_path = impl_->opts.model_path;
    eopts.n_ctx = impl_->opts.n_ctx;
    eopts.n_threads = impl_->opts.n_threads;
    eopts.n_gpu_layers = impl_->opts.n_gpu_layers;
    eopts.seed = impl_->opts.seed;
    engine = std::make_unique<minisgl::engine::LlamaCppEngine>(eopts);
#else
    throw std::runtime_error("llama_cpp backend requested but llama.cpp headers were not found at build time");
#endif
  } else {
    throw std::runtime_error("Unknown backend: " + impl_->opts.backend);
  }

  impl_->svc = std::make_unique<InferenceServiceImpl>(std::move(engine));
  builder.RegisterService(impl_->svc.get());

  impl_->server = builder.BuildAndStart();
  if (!impl_->server) {
    throw std::runtime_error("Failed to start gRPC server");
  }

  std::cerr << "minisgl_grpc_server listening on " << impl_->opts.listen_addr << "\n";
  impl_->server->Wait();
}

} // namespace minisgl


