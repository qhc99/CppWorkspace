cmake_minimum_required(VERSION 3.28)

find_package(fmt 10 CONFIG REQUIRED)

set(folder_name central_lib)
set(LIB_SOURCES
    DisjointSet.cpp
    utils.cpp
    numeric_utils.cpp
    public/lib_central/concepts_utils.h
    public/lib_central/DisjointSet.h
    public/lib_central/numeric_utils.h
    public/lib_central/rank_search.hpp
    public/lib_central/TernaryTries.hpp
    public/lib_central/utils.h
    )

add_library(central_lib STATIC ${LIB_SOURCES})
target_include_directories(central_lib PUBLIC public)
set_target_properties(central_lib PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(central_lib PRIVATE ${COMMON_LINK_OPTIONS})
target_link_libraries(central_lib PRIVATE workspace_pch)


add_library(central_lib_shared SHARED ${LIB_SOURCES})
target_include_directories(central_lib_shared PRIVATE 
                            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>"
                           "$<INSTALL_INTERFACE:include>") # Required for install
set_target_properties(central_lib_shared PROPERTIES FOLDER ${folder_name})
target_compile_options(central_lib_shared PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(central_lib_shared PRIVATE ${COMMON_LINK_OPTIONS})
target_link_libraries(central_lib_shared PRIVATE workspace_pch)
target_compile_definitions(central_lib_shared PRIVATE BUILDING_CENTRAL_LIB USE_SHARED)

# Install
install(TARGETS central_lib_shared
        EXPORT CentralLibTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)
install(DIRECTORY public/lib_central DESTINATION include)
install(EXPORT CentralLibTargets
        FILE CentralLibTargets.cmake # consumed in config.cmake.in
        NAMESPACE CentralLib::
        DESTINATION lib/cmake/central_lib
)

# Create package from install

include(CMakePackageConfigHelpers)

set(CENTRAL_LIB_VERSION 1.0.0)
set_property(TARGET central_lib_shared PROPERTY VERSION ${CENTRAL_LIB_VERSION})
set_property(TARGET central_lib_shared PROPERTY SOVERSION 1)
set_property(TARGET central_lib_shared PROPERTY
  INTERFACE_central_lib_shared_MAJOR_VERSION 1)
set_property(TARGET central_lib_shared APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING central_lib_shared_MAJOR_VERSION
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CentralLibConfigVersion.cmake"
  VERSION "${CENTRAL_LIB_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/CentralLibConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CentralLibConfig.cmake"
  INSTALL_DESTINATION lib/cmake/central_lib
)

install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/CentralLibConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/CentralLibConfigVersion.cmake"
        DESTINATION lib/cmake/central_lib
)


if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_library(central_lib_asan STATIC ${LIB_SOURCES})
    target_include_directories(central_lib_asan PUBLIC public)
    set_target_properties(central_lib_asan PROPERTIES FOLDER ${folder_name})
    target_compile_options(central_lib_asan PRIVATE ${ASAN_COMPILE_OPTIONS} ${COMMON_COMPILE_OPTIONS})
    target_link_options(central_lib_asan PRIVATE ${ASAN_LINK_OPTIONS} ${COMMON_LINK_OPTIONS})
    target_link_libraries(central_lib_asan PRIVATE workspace_pch)
endif()

add_executable(
    lib_main
    main.cpp)
target_link_libraries(lib_main PRIVATE central_lib fmt::fmt-header-only workspace_pch)
set_target_properties(lib_main PROPERTIES FOLDER ${folder_name})