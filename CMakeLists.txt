cmake_minimum_required(VERSION 3.2)

project(gameai-optflow)
include_directories(include)
find_package(OpenCV REQUIRED)
add_executable(gameai-optflow src/main.cpp)
set_property(TARGET gameai-optflow PROPERTY CXX_STANDARD 11)
set_property(TARGET gameai-optflow PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(gameai-optflow ${OpenCV_LIBS})
