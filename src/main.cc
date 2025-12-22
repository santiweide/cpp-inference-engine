#include "server.h"

#include <iostream>
#include <string>

namespace {

minisgl::Server::Options parse_args(int argc, char **argv) {
  minisgl::Server::Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--listen" && i + 1 < argc) {
      opts.listen_addr = argv[++i];
    } else if (a == "--backend" && i + 1 < argc) {
      opts.backend = argv[++i];
    } else if (a == "--model_path" && i + 1 < argc) {
      opts.model_path = argv[++i];
    } else if (a == "--n_ctx" && i + 1 < argc) {
      opts.n_ctx = std::stoi(argv[++i]);
    } else if (a == "--n_threads" && i + 1 < argc) {
      opts.n_threads = std::stoi(argv[++i]);
    } else if (a == "--n_gpu_layers" && i + 1 < argc) {
      opts.n_gpu_layers = std::stoi(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      opts.seed = std::stoi(argv[++i]);
    } else if (a == "-h" || a == "--help") {
      std::cout
          << "Usage: minisgl_grpc_server [--listen host:port]\n"
          << "                         [--backend dummy|llama_cpp]\n"
          << "                         [--model_path /path/to/model.gguf]\n"
          << "                         [--n_ctx 4096] [--n_threads N]\n"
          << "                         [--n_gpu_layers N] [--seed N]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      std::exit(2);
    }
  }
  return opts;
}

} // namespace

int main(int argc, char **argv) {
  try {
    auto opts = parse_args(argc, argv);
    minisgl::Server server(opts);
    server.RunBlocking();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
}


