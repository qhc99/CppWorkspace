cmake_minimum_required(VERSION 3.16)

project(algo_cpp)

add_library(${PROJECT_NAME}
        SHARED
        DisjointSet.cpp
        )
add_library(org::nathan::algo_cpp ALIAS ${PROJECT_NAME})
include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(${PROJECT_NAME}           # generates the export header shared_EXPORTS.h automatically
        BASE_NAME ${PROJECT_NAME}
        EXPORT_MACRO_NAME ${PROJECT_NAME}_EXPORTS
        EXPORT_FILE_NAME ${PROJECT_NAME}_EXPORTS.h
        STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC include)
