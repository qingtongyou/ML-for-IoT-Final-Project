cmake_minimum_required(VERSION 4.1)
project(mfe_extract_cpp)

set(CMAKE_CXX_STANDARD 20)

add_executable(mfe_extract_cpp main.cpp
        mfe.cpp)
