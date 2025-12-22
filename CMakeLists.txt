cmake_minimum_required(VERSION 3.20)
project(minisgl_grpc_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINISGL_SERVER_BUILD_TESTS "Build tests" OFF)
option(MINISGL_ENABLE_LLAMA_CPP "Enable llama.cpp backend" ON)
set(LLAMA_CPP_DIR "" CACHE PATH "Path to a local llama.cpp checkout (required if MINISGL_ENABLE_LLAMA_CPP=ON)")

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PROTO_FILE ${PROTO_DIR}/minisgl_inference.proto)

add_library(minisgl_proto STATIC
  ${PROTO_FILE}
)

protobuf_generate(
  TARGET minisgl_proto
  LANGUAGE cpp
  PROTOS ${PROTO_FILE}
  OUT_VAR PROTO_SRCS
  PROTOC_OUT_DIR ${GEN_DIR}
)

protobuf_generate(
  TARGET minisgl_proto
  LANGUAGE grpc
  GENERATE_EXTENSIONS .grpc.pb.cc .grpc.pb.h
  PROTOS ${PROTO_FILE}
  OUT_VAR GRPC_SRCS
  PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
  PROTOC_OUT_DIR ${GEN_DIR}
)

target_sources(minisgl_proto PRIVATE ${PROTO_SRCS} ${GRPC_SRCS})
target_include_directories(minisgl_proto PUBLIC ${GEN_DIR})
target_link_libraries(minisgl_proto PUBLIC protobuf::libprotobuf gRPC::grpc++ gRPC::grpc)

add_executable(minisgl_grpc_server
  src/main.cc
  src/server.cc
  src/engine/dummy_engine.cc
)

target_include_directories(minisgl_grpc_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(minisgl_grpc_server PRIVATE minisgl_proto)

if (MINISGL_ENABLE_LLAMA_CPP)
  if (LLAMA_CPP_DIR STREQUAL "")
    message(FATAL_ERROR "MINISGL_ENABLE_LLAMA_CPP=ON but LLAMA_CPP_DIR is empty. Set -DLLAMA_CPP_DIR=/path/to/llama.cpp")
  endif()
  if (NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR does not look like a llama.cpp checkout: ${LLAMA_CPP_DIR}")
  endif()

  # Build llama.cpp as a subproject. CUDA must be enabled in llama.cpp itself.
  add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama_cpp EXCLUDE_FROM_ALL)

  if (TARGET llama)
    target_link_libraries(minisgl_grpc_server PRIVATE llama)
  else()
    message(FATAL_ERROR "llama.cpp did not define target 'llama'. Please check your llama.cpp version.")
  endif()

  # Common include location for llama.cpp is ${LLAMA_CPP_DIR}/include
  if (EXISTS "${LLAMA_CPP_DIR}/include")
    target_include_directories(minisgl_grpc_server PRIVATE ${LLAMA_CPP_DIR}/include)
  else()
    target_include_directories(minisgl_grpc_server PRIVATE ${LLAMA_CPP_DIR})
  endif()
  target_sources(minisgl_grpc_server PRIVATE src/engine/llama_cpp_engine.cc)
else()
  message(STATUS "MINISGL_ENABLE_LLAMA_CPP=OFF: building server without llama.cpp backend")
endif()


