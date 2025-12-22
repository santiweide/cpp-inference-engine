# mini-sglang C++ Inference Server (gRPC, single-node)

This is a **pure C++** inference server (no Python runtime) that exposes a **gRPC** API.

It is intentionally a **skeleton**:
- The gRPC surface and server runtime are implemented.
- The inference backend is an **Engine** interface with a **DummyEngine** implementation.
- Plug in a real backend later (e.g. llama.cpp/ggml, TensorRT-LLM, libtorch CUDA).

## Directory layout

- `proto/minisgl_inference.proto`: gRPC API definition
- `src/`: server + engine interface + dummy backend
- `CMakeLists.txt`: build (expects gRPC + Protobuf installed, or use vcpkg)

## Build

### Option A: Use system-installed gRPC + Protobuf

Requirements (typical):
- CMake >= 3.20
- a C++20 compiler
- Protobuf + gRPC development packages

Build:

```bash
cd cpp_inference_server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Option B: Use vcpkg

Install packages:
- `grpc`
- `protobuf`

Then:

```bash
cd cpp_inference_server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build -j
```

## Run

```bash
./cpp_inference_server/build/minisgl_grpc_server --listen 0.0.0.0:50051
```

## Call with grpcurl

Example:

```bash
grpcurl -plaintext -d '{"model":"dummy","prompt":"Hello","max_tokens":16}' \
  127.0.0.1:50051 minisgl.inference.InferenceService/Generate
```

## Use llama.cpp (CUDA)

This server can call **llama.cpp** via its C API. CUDA support comes from building llama.cpp with
`GGML_CUDA=ON` (or your llama.cpp equivalent).

### Build (example)

Assuming you have a local checkout of llama.cpp at `/path/to/llama.cpp`:

```bash
cd cpp_inference_server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DMINISGL_ENABLE_LLAMA_CPP=ON
cmake --build build -j
```

Then run:

```bash
./build/minisgl_grpc_server \
  --backend llama_cpp \
  --model_path /path/to/model.gguf \
  --listen 0.0.0.0:50051 \
  --n_ctx 4096 \
  --n_gpu_layers 999 \
  --n_threads 8
```

Call:

```bash
grpcurl -plaintext -d '{"model":"llama","prompt":"Hello","max_tokens":64}' \
  127.0.0.1:50051 minisgl.inference.InferenceService/Generate
```



