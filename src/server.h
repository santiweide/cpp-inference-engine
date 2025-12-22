#pragma once

#include <memory>
#include <string>

namespace minisgl {

class Server {
public:
  struct Options {
    std::string listen_addr = "0.0.0.0:50051";
    std::string backend = "dummy"; // dummy | llama_cpp
    std::string model_path;        // required for llama_cpp
    int n_ctx = 4096;
    int n_threads = 0;
    int n_gpu_layers = -1;
    int seed = -1;
  };

  explicit Server(Options opts);
  ~Server();

  void RunBlocking();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace minisgl


