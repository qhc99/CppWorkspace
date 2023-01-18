cmake_minimum_required(VERSION 3.20)

add_library(lib_central
        STATIC
        DisjointSet.cpp
        utils.cpp
        numerics.cpp
        public/lib_central/DisjointSet.h
        public/lib_central/numerics.h
        public/lib_central/utils.h
        public/lib_central/rank_search.hpp
        public/lib_central/TernaryTries.hpp)
target_include_directories(lib_central PUBLIC public)

add_executable(
        lib_main
        main.cpp)
target_link_libraries(lib_main PUBLIC lib_central)

