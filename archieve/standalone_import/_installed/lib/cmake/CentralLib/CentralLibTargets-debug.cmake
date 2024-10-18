#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CentralLib::central_lib_shared" for configuration "Debug"
set_property(TARGET CentralLib::central_lib_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(CentralLib::central_lib_shared PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/central_lib_shared.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/central_lib_shared.dll"
  )

list(APPEND _cmake_import_check_targets CentralLib::central_lib_shared )
list(APPEND _cmake_import_check_files_for_CentralLib::central_lib_shared "${_IMPORT_PREFIX}/lib/central_lib_shared.lib" "${_IMPORT_PREFIX}/bin/central_lib_shared.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
