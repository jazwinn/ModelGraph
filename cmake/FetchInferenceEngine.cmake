# Keeps a prebuilt InferenceEngine release in IE_PREFIX up to date.
#
# Used two ways:
#   include(FetchInferenceEngine.cmake) + ie_sync()   at configure time
#   cmake -D... -P FetchInferenceEngine.cmake         before every build
#
# Inputs (variables):
#   IE_REPO            GitHub "owner/name"
#   IE_VERSION         "latest" or a release tag such as v0.1.0
#   IE_VARIANT         cpu | cuda
#   IE_PREFIX          where the package is unpacked
#   IE_UPDATED_STAMP   file touched after an update (makes CMake reconfigure)
#   IE_API_URL         optional: override the "latest release" API URL
#   IE_DOWNLOAD_BASE   optional: override https://github.com/<repo>/releases/download
#
# Offline or rate-limited? The installed copy keeps working; only a missing
# package is an error.

function(_ie_latest_tag out)
    set(${out} "" PARENT_SCOPE)
    if(IE_API_URL)
        set(url "${IE_API_URL}")
    else()
        set(url "https://api.github.com/repos/${IE_REPO}/releases/latest")
    endif()
    set(headers HTTPHEADER "Accept: application/vnd.github+json" HTTPHEADER "User-Agent: ModelGraph-build")
    if(DEFINED ENV{GITHUB_TOKEN} AND NOT "$ENV{GITHUB_TOKEN}" STREQUAL "")
        list(APPEND headers HTTPHEADER "Authorization: Bearer $ENV{GITHUB_TOKEN}")
    endif()
    set(json_file "${IE_PREFIX}-latest.json")
    file(DOWNLOAD "${url}" "${json_file}" STATUS status TIMEOUT 20 ${headers})
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        list(GET status 1 reason)
        message(STATUS "InferenceEngine: could not check for releases (${reason})")
        return()
    endif()
    file(READ "${json_file}" json)
    string(JSON tag ERROR_VARIABLE err GET "${json}" tag_name)
    if(err OR tag STREQUAL "")
        message(STATUS "InferenceEngine: no published release found at ${url}")
        return()
    endif()
    set(${out} "${tag}" PARENT_SCOPE)
endfunction()

function(ie_sync)
    foreach(var IE_REPO IE_VERSION IE_VARIANT IE_PREFIX)
        if(NOT ${var})
            message(FATAL_ERROR "FetchInferenceEngine: ${var} is not set")
        endif()
    endforeach()

    set(stamp_file "${IE_PREFIX}/MODELGRAPH_IE_VERSION.txt")
    set(installed "")
    if(EXISTS "${stamp_file}")
        file(READ "${stamp_file}" installed)
        string(STRIP "${installed}" installed)
    endif()

    if(IE_VERSION STREQUAL "latest")
        _ie_latest_tag(tag)
    else()
        set(tag "${IE_VERSION}")
    endif()

    if(tag STREQUAL "")
        if(installed)
            message(STATUS "InferenceEngine: using installed ${installed}")
            return()
        endif()
        message(FATAL_ERROR
            "InferenceEngine: no release is installed and none could be downloaded from ${IE_REPO}.\n"
            "Check your connection, pin a version with -DIE_VERSION=<tag>, or build from a local "
            "checkout with -DIE_SOURCE_DIR=<path to InferenceEngine>.")
    endif()

    set(wanted "${tag} ${IE_VARIANT}")
    if(installed STREQUAL wanted)
        message(STATUS "InferenceEngine: ${wanted} is up to date")
        return()
    endif()

    set(asset "InferenceEngine-${tag}-windows-x64-${IE_VARIANT}.zip")
    if(IE_DOWNLOAD_BASE)
        set(url "${IE_DOWNLOAD_BASE}/${tag}/${asset}")
    else()
        set(url "https://github.com/${IE_REPO}/releases/download/${tag}/${asset}")
    endif()
    if(installed)
        message(STATUS "InferenceEngine: updating ${installed} -> ${wanted}")
    else()
        message(STATUS "InferenceEngine: downloading ${wanted}")
    endif()

    set(zip "${IE_PREFIX}-download.zip")
    file(DOWNLOAD "${url}" "${zip}" STATUS status SHOW_PROGRESS TIMEOUT 600)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        list(GET status 1 reason)
        file(REMOVE "${zip}")
        if(installed)
            message(WARNING "InferenceEngine: download of ${url} failed (${reason}); keeping ${installed}")
            return()
        endif()
        message(FATAL_ERROR "InferenceEngine: download of ${url} failed (${reason})")
    endif()

    # Unpack beside the old copy, then swap, so a failure never leaves a half package.
    set(staging "${IE_PREFIX}-staging")
    file(REMOVE_RECURSE "${staging}")
    file(ARCHIVE_EXTRACT INPUT "${zip}" DESTINATION "${staging}")
    file(REMOVE "${zip}")
    set(root "${staging}")
    if(NOT EXISTS "${staging}/lib/cmake/InferenceEngine")
        file(GLOB entries LIST_DIRECTORIES true "${staging}/*")
        list(LENGTH entries count)
        if(count EQUAL 1 AND IS_DIRECTORY "${entries}")
            set(root "${entries}")
        endif()
    endif()
    if(NOT EXISTS "${root}/lib/cmake/InferenceEngine/InferenceEngineConfig.cmake")
        file(REMOVE_RECURSE "${staging}")
        message(FATAL_ERROR "InferenceEngine: ${asset} does not contain lib/cmake/InferenceEngine")
    endif()
    file(REMOVE_RECURSE "${IE_PREFIX}")
    file(RENAME "${root}" "${IE_PREFIX}")
    file(REMOVE_RECURSE "${staging}")
    file(WRITE "${stamp_file}" "${wanted}\n")

    if(IE_UPDATED_STAMP)
        file(TOUCH "${IE_UPDATED_STAMP}")
    endif()
    message(STATUS "InferenceEngine: installed ${wanted}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
    ie_sync()
endif()
