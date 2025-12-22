#pragma once

#include "engine/engine.h"

namespace minisgl::engine {

class DummyEngine final : public Engine {
public:
  GenerateResult Generate(const std::string &model,
                          const std::string &prompt,
                          const SamplingParams &params,
                          const std::string &session_id) override;
};

} // namespace minisgl::engine


