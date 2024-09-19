#ifndef DEV_QC_CENTRAL_LIB_DLL_H
#define DEV_QC_CENTRAL_LIB_DLL_H


#ifdef USE_SHARED
  #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef BUILDING_CENTRAL_LIB
      #define CENTRAL_LIB_API __declspec(dllexport)
    #else
      #define CENTRAL_LIB_API __declspec(dllimport)
    #endif
  #else
    #ifdef BUILDING_CENTRAL_LIB
      #define CENTRAL_LIB_API __attribute__((visibility("default")))
    #else
      #define CENTRAL_LIB_API
    #endif
  #endif
#else
  #define CENTRAL_LIB_API
#endif

#include <vector>

namespace dev::qhc::utils {

using std::vector;

CENTRAL_LIB_API vector<int> shuffledRange(int low, int high);

} // namespace dev::qhc::utils


#endif