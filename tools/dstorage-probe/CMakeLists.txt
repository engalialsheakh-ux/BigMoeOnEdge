cmake_minimum_required(VERSION 3.21)
project(bmoe_dstorage_probe LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
if(NOT DSTORAGE_NATIVE)
    message(FATAL_ERROR "Set DSTORAGE_NATIVE to the DirectStorage SDK package's native directory")
endif()

add_executable(bmoe_dstorage_probe main.cpp)
target_compile_features(bmoe_dstorage_probe PRIVATE cxx_std_17)
target_include_directories(bmoe_dstorage_probe PRIVATE "${DSTORAGE_NATIVE}/include")
target_link_libraries(bmoe_dstorage_probe PRIVATE
    "${DSTORAGE_NATIVE}/lib/x64/dstorage.lib" d3d12 dxgi CUDA::cudart)
add_custom_command(TARGET bmoe_dstorage_probe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DSTORAGE_NATIVE}/bin/x64/dstorage.dll"
        "${DSTORAGE_NATIVE}/bin/x64/dstoragecore.dll"
        "$<TARGET_FILE_DIR:bmoe_dstorage_probe>")
