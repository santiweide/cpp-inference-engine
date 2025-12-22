#include "engine/dummy_engine.h"

#include <sstream>

namespace minisgl::engine {

GenerateResult DummyEngine::Generate(const std::string &model,
                                     const std::string &prompt,
                                     const SamplingParams &params,
                                     const std::string &session_id) {
  (void)params;
  std::ostringstream oss;
  oss << "[dummy model=" << model << " session_id=" << session_id << "] ";
  oss << "echo: " << prompt;
  GenerateResult r;
  r.text = oss.str();
  r.prompt_tokens = static_cast<int>(prompt.size());
  r.completion_tokens = static_cast<int>(r.text.size());
  return r;
}

} // namespace minisgl::engine


