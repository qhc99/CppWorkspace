cmake_minimum_required(VERSION 3.28)

project(workspace_pch LANGUAGES CXX)
add_library(workspace_pch INTERFACE)
target_precompile_headers(workspace_pch INTERFACE include/workspace_pch.h)
target_include_directories(workspace_pch INTERFACE include)
