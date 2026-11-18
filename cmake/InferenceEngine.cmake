# Where InferenceEngine comes from:
#
#   default            the latest GitHub release of IE_REPO, downloaded into the
#                      build tree and re-checked before every build
#   -DIE_VERSION=v0.2  a fixed release instead of the latest
#   -DIE_SOURCE_DIR=.. a local checkout, built from source (for working on both)

set(IE_SOURCE_DIR "" CACHE PATH "Build InferenceEngine from this checkout; empty = use a GitHub release")
set(IE_VERSION "latest" CACHE STRING "InferenceEngine release to use: latest, or a tag such as v0.1.0")
set(IE_REPO "jazwinn/InferenceEngine" CACHE STRING "GitHub repository that publishes InferenceEngine releases")
set(IE_API_URL "" CACHE STRING "Override the GitHub 'latest release' API URL (mirrors, testing)")
set(IE_DOWNLOAD_BASE "" CACHE STRING "Override the release download base URL (mirrors, testing)")
mark_as_advanced(IE_REPO IE_API_URL IE_DOWNLOAD_BASE)

if(IE_SOURCE_DIR)
    if(NOT EXISTS "${IE_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "IE_SOURCE_DIR='${IE_SOURCE_DIR}' is not an InferenceEngine checkout")
    endif()
    message(STATUS "InferenceEngine: building from source at ${IE_SOURCE_DIR}")
    if(IE_ENABLE_CUDA)
        enable_language(CUDA)
    endif()
    add_subdirectory(${IE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/InferenceEngine EXCLUDE_FROM_ALL)
    return()
endif()

if(IE_ENABLE_CUDA)
    set(IE_VARIANT cuda)
else()
    set(IE_VARIANT cpu)
endif()
set(IE_PREFIX "${CMAKE_BINARY_DIR}/_deps/inference-engine-${IE_VARIANT}")
set(IE_UPDATED_STAMP "${CMAKE_BINARY_DIR}/_deps/inference-engine-updated.stamp")

include(${CMAKE_CURRENT_LIST_DIR}/FetchInferenceEngine.cmake)
ie_sync()

# A release downloaded during a build touches this stamp, so the next build
# reconfigures and picks up any change to the package's targets.
if(NOT EXISTS "${IE_UPDATED_STAMP}")
    file(TOUCH "${IE_UPDATED_STAMP}")
endif()
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${IE_UPDATED_STAMP}")

find_package(InferenceEngine CONFIG REQUIRED PATHS "${IE_PREFIX}" NO_DEFAULT_PATH)
if((InferenceEngine_CUDA AND NOT IE_ENABLE_CUDA) OR (NOT InferenceEngine_CUDA AND IE_ENABLE_CUDA))
    message(FATAL_ERROR "The downloaded InferenceEngine (${IE_VARIANT}) does not match IE_ENABLE_CUDA=${IE_ENABLE_CUDA}")
endif()

# Before every build: check for a newer release and install it.
add_custom_target(inference_engine_update ALL
    COMMAND ${CMAKE_COMMAND}
        -DIE_REPO=${IE_REPO}
        -DIE_VERSION=${IE_VERSION}
        -DIE_VARIANT=${IE_VARIANT}
        -DIE_PREFIX=${IE_PREFIX}
        -DIE_UPDATED_STAMP=${IE_UPDATED_STAMP}
        -DIE_API_URL=${IE_API_URL}
        -DIE_DOWNLOAD_BASE=${IE_DOWNLOAD_BASE}
        -P ${CMAKE_CURRENT_LIST_DIR}/FetchInferenceEngine.cmake
    COMMENT "Checking for InferenceEngine updates"
    VERBATIM
)
