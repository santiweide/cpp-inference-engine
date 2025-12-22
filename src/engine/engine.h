#pragma once

#include <string>

namespace minisgl::engine {

struct SamplingParams {
  int max_tokens = 128;
  float temperature = 0.0f;
  int top_k = 1;
  bool ignore_eos = false;
};

struct GenerateResult {
  std::string text;
  int prompt_tokens = 0;
  int completion_tokens = 0;
};

class Engine {
public:
  virtual ~Engine() = default;

  virtual GenerateResult Generate(const std::string &model,
                                  const std::string &prompt,
                                  const SamplingParams &params,
                                  const std::string &session_id) = 0;
};

} // namespace minisgl::engine


