# Locates the vcpkg toolchain file and wires it in before project() runs, which
# is what lets manifest mode (vcpkg.json) install dependencies during configure.
#
# Resolution order:
#   1. An explicit -DCMAKE_TOOLCHAIN_FILE=... always wins.
#   2. $ENV{VCPKG_ROOT}
#   3. Common install locations.

if(DEFINED CMAKE_TOOLCHAIN_FILE)
    return()
endif()

set(_vcpkg_candidates "")

if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_env_root)
    list(APPEND _vcpkg_candidates "${_vcpkg_env_root}")
endif()

foreach(_home IN ITEMS "$ENV{USERPROFILE}" "$ENV{HOME}")
    if(_home)
        file(TO_CMAKE_PATH "${_home}" _home_path)
        list(APPEND _vcpkg_candidates "${_home_path}/vcpkg")
    endif()
endforeach()

list(APPEND _vcpkg_candidates "C:/vcpkg" "/opt/vcpkg" "/usr/local/vcpkg")

foreach(_root IN LISTS _vcpkg_candidates)
    if(EXISTS "${_root}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE "${_root}/scripts/buildsystems/vcpkg.cmake"
            CACHE FILEPATH "vcpkg toolchain file")
        message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
        return()
    endif()
endforeach()

message(FATAL_ERROR
    "Could not locate vcpkg. Set VCPKG_ROOT, or configure with "
    "-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake")
