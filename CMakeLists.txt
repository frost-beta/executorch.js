# ----------------------------- Setup ----------------------------------

cmake_minimum_required(VERSION 3.24)
include(FetchContent)

# https://github.com/cmake-js/cmake-js#node-api-and-node-addon-api
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
# Allow FetchContent_Populate.
cmake_policy(SET CMP0169 OLD)

# ----------------------------- Project --------------------------------

# Define project name.
project(executorch.js)
# Use C++20.
set(CMAKE_CXX_STANDARD 20)

# ----------------------------- Config ---------------------------------

set(TORCH_VERSION "0.0.4" CACHE STRING "The version of executorch-binaries")
set(TORCH_BINARY_URL "https://github.com/frost-beta/executorch-binaries/releases/download" CACHE STRING "The download URL of executorch-binaries")
option(TORCH_BACKEND_ALL "Build with all backends" OFF)
option(TORCH_BACKEND_COREML "Build with CoreML backend" OFF)
option(TORCH_BACKEND_MPS "Build with MPS backend" OFF)
option(TORCH_BACKEND_XNNPACK "Build with XNNPACK backend" OFF)
option(TORCH_KERNELS_CUSTOM "Build with custom kernels" ON)
option(TORCH_KERNELS_OPTIMIZED "Build with optimzied kernels" ON)
option(TORCH_KERNELS_QUANTIZED "Build with quantized kernels" ON)

if(TORCH_BACKEND_ALL)
  set(TORCH_BACKEND_XNNPACK ON)
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(TORCH_BACKEND_MPS ON)
  endif()
endif()

# https://github.com/pytorch/executorch/issues/6839
if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(TORCH_KERNELS_CUSTOM OFF)
  set(TORCH_KERNELS_OPTIMIZED OFF)
endif()

# ----------------------------- Node-API -------------------------------

# Define NAPI_VERSION.
add_definitions(-DNAPI_VERSION=9)

# Use node headers/libs provided by cmake-js..
add_library(${PROJECT_NAME} SHARED ${CMAKE_JS_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      PREFIX ""
                      OUTPUT_NAME "executorch"
                      SUFFIX ".node")
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR}
                  /def:${CMAKE_JS_NODELIB_DEF}
                  /out:${CMAKE_JS_NODELIB_TARGET}
                  ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# -------------------------- Utilities ---------------------------------

# Behave the same with the target_link_options_shared_lib util of ExecuTorch.
function(target_force_link_libraries target_name linker_type)
  foreach(target_lib ${ARGN})
    if(APPLE)
      target_link_options(${target_name} ${linker_type}
                          "SHELL:LINKER:-force_load,${target_lib}")
    elseif(MSVC)
      target_link_options(${target_name} ${linker_type}
                          "SHELL:LINKER:/WHOLEARCHIVE:${target_lib}")
    else()
      target_link_options(${target_name} ${linker_type}
                          "SHELL:LINKER:--whole-archive \
                          ${target_lib} \
                          LINKER:--no-whole-archive")
    endif()
  endforeach()
endfunction()

# ----------------------------- Lib ------------------------------------

# Add source files.
file(GLOB SOURCE_FILES "src/*")
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ".")

# Use kizunapi.
FetchContent_Declare(kizunapi GIT_REPOSITORY https://github.com/yue/kizunapi.git
                              GIT_TAG 4c59c84e6dd1d2da260b257adf1bd8c5244fde32)
FetchContent_MakeAvailable(kizunapi)
target_include_directories(${PROJECT_NAME} PRIVATE "${kizunapi_SOURCE_DIR}")

# Use fmt when building with old compilers.
target_compile_definitions(${PROJECT_NAME} PRIVATE ETJS_USE_FMT)
FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt
                         GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281)
# Do not call FetchContent_MakeAvailable as we only need headers.
FetchContent_GetProperties(fmt)
if(NOT fmt_POPULATED)
  FetchContent_Populate(fmt)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE "${fmt_SOURCE_DIR}/include")

# Download the executorch binaries.
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(TORCH_OS "mac")
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(TORCH_OS "linux")
else()
  message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(TORCH_ARCH "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  set(TORCH_ARCH "arm64")
else()
  message(FATAL_ERROR "Unsupported arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(TORCH_CONFIG "debug")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ETJS_DEBUG_BUILD)
else()
  set(TORCH_CONFIG "release")
endif()
FetchContent_Declare(torch_lib URL "${TORCH_BINARY_URL}/v${TORCH_VERSION}/executorch-full-${TORCH_CONFIG}-${TORCH_OS}-${TORCH_ARCH}.zip")
FetchContent_MakeAvailable(torch_lib)
set(TORCH_LIBS "${torch_lib_SOURCE_DIR}/libs")

# Link with correct libs.
target_include_directories(${PROJECT_NAME} PRIVATE
                           "${torch_lib_SOURCE_DIR}/include")
target_link_libraries(${PROJECT_NAME} PRIVATE
                      "${TORCH_LIBS}/libcpuinfo.a"
                      "${TORCH_LIBS}/libexecutorch_core.a"
                      "${TORCH_LIBS}/libextension_module_static.a")
target_force_link_libraries(${PROJECT_NAME} PRIVATE
                            "${TORCH_LIBS}/libexecutorch.a"
                            "${TORCH_LIBS}/libextension_data_loader.a")
if(TORCH_BACKEND_COREML)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ETJS_BACKEND_COREML)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  find_library(COREML_FRAMEWORK CoreML)
  target_link_libraries(${PROJECT_NAME} PUBLIC
                        ${ACCELERATE_FRAMEWORK}
                        ${COREML_FRAMEWORK})
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/libcoremldelegate.a")
endif()
if(TORCH_BACKEND_MPS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ETJS_BACKEND_MPS)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(METAL_FRAMEWORK Metal)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  find_library(MPS_GRAPH_FRAMEWORK MetalPerformanceShadersGraph)
  target_link_libraries(${PROJECT_NAME} PUBLIC
                        ${FOUNDATION_FRAMEWORK}
                        ${METAL_FRAMEWORK}
                        ${MPS_FRAMEWORK}
                        ${MPS_GRAPH_FRAMEWORK})
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/libmpsdelegate.a")
endif()
if(TORCH_BACKEND_XNNPACK)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ETJS_BACKEND_XNNPACK)
  target_link_libraries(${PROJECT_NAME} PRIVATE
                        "${TORCH_LIBS}/libpthreadpool.a")
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/libXNNPACK.a"
                              "${TORCH_LIBS}/libmicrokernels-prod.a"
                              "${TORCH_LIBS}/libxnnpack_backend.a")
endif()
if(TORCH_KERNELS_CUSTOM)
  target_link_libraries(${PROJECT_NAME} PRIVATE
                        "${TORCH_LIBS}/libcpublas.a"
                        "${TORCH_LIBS}/libeigen_blas.a"
                        "${TORCH_LIBS}/libportable_kernels.a"
                        "${TORCH_LIBS}/libpthreadpool.a")
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/libcustom_ops.a")
else()
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/libportable_kernels.a"
                              "${TORCH_LIBS}/libportable_ops_lib.a")
endif()
if(TORCH_KERNELS_OPTIMIZED)
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/liboptimized_kernels.a"
                              "${TORCH_LIBS}/liboptimized_native_cpu_ops_lib.a")
endif()
if(TORCH_KERNELS_QUANTIZED)
  target_force_link_libraries(${PROJECT_NAME} PRIVATE
                              "${TORCH_LIBS}/libquantized_kernels.a"
                              "${TORCH_LIBS}/libquantized_ops_lib.a")
endif()

# --------------------------- Summary ----------------------------------

message(STATUS "")
message(STATUS "******** Summary ********")
message(STATUS "  CMAKE_BUILD_TYPE              : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD            : ${CMAKE_CXX_STANDARD}")
message(STATUS "  TORCH_VERSION                 : ${TORCH_VERSION}")
message(STATUS "  TORCH_BINARY_URL              : ${TORCH_BINARY_URL}")
message(STATUS "  TORCH_BACKEND_ALL             : ${TORCH_BACKEND_ALL}")
message(STATUS "  TORCH_BACKEND_COREML          : ${TORCH_BACKEND_COREML}")
message(STATUS "  TORCH_BACKEND_MPS             : ${TORCH_BACKEND_MPS}")
message(STATUS "  TORCH_BACKEND_XNNPACK         : ${TORCH_BACKEND_XNNPACK}")
message(STATUS "  TORCH_KERNELS_CUSTOM          : ${TORCH_KERNELS_CUSTOM}")
message(STATUS "  TORCH_KERNELS_OPTIMIZED       : ${TORCH_KERNELS_OPTIMIZED}")
message(STATUS "  TORCH_KERNELS_QUANTIZED       : ${TORCH_KERNELS_QUANTIZED}")
message(STATUS "")
