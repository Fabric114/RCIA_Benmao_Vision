cmake_minimum_required(VERSION 3.10)

project(RCIA_Energy_trace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/m3t)

add_executable(run_m3t_tracking
    m3t_app/run_m3t_tracking.cpp
)

target_link_libraries(run_m3t_tracking PRIVATE m3t)
target_compile_definitions(run_m3t_tracking PRIVATE
    RCIA_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
