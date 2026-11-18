# ---- Graph: IR, shape inference, compiler (no UI, no scripting) -------------
add_library(mg_graph STATIC
    Graph/Expr.cpp
    Graph/Document.cpp
    Graph/Registry.cpp
    Graph/Analyze.cpp
    Graph/Compiler.cpp
    Graph/Builtins.cpp
    Graph/Weights.cpp
    Graph/Templates.cpp
)
add_library(ModelGraph::Graph ALIAS mg_graph)
target_include_directories(mg_graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mg_graph PUBLIC
    InferenceEngine::NeuralNet
    InferenceEngine::Models
    InferenceEngine::Serialize
    InferenceEngine::Generation
)
target_compile_options(mg_graph PRIVATE ${MG_WARNINGS})
if(TARGET inference_engine_update)
    add_dependencies(mg_graph inference_engine_update)   # update the engine before compiling against it
endif()

# ---- Script: Python node definitions with hot reload -------------------------
# Embeds the Python found on this machine (override with -DPython_ROOT_DIR=...).
find_package(Python 3.9 COMPONENTS Interpreter Development.Embed REQUIRED)
message(STATUS "Embedding Python ${Python_VERSION} from ${Python_EXECUTABLE}")

include(FetchContent)
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(pybind11)

add_library(mg_script STATIC
    Script/PyNodes.cpp
    Script/ScriptHost.cpp
)
add_library(ModelGraph::Script ALIAS mg_script)
target_include_directories(mg_script PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mg_script PUBLIC mg_graph pybind11::embed)
cmake_path(GET Python_EXECUTABLE PARENT_PATH MG_PYTHON_HOME)
target_compile_definitions(mg_script PRIVATE MG_PYTHON_HOME="${MG_PYTHON_HOME}")
target_compile_options(mg_script PRIVATE ${MG_WARNINGS})

# Executables that embed Python need its DLL beside them.
file(GLOB _mg_python_dlls "${MG_PYTHON_HOME}/python3*.dll")
set(MG_PYTHON_DLLS "${_mg_python_dlls}" CACHE INTERNAL "Python DLLs copied next to executables")
function(mg_copy_python target)
    foreach(dll IN LISTS MG_PYTHON_DLLS)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:${target}>")
    endforeach()
endfunction()

# ---- Session: build / load / generate on a worker thread ---------------------
add_library(mg_session STATIC Session/Session.cpp)
target_include_directories(mg_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mg_session PUBLIC mg_graph mg_script)
target_compile_options(mg_session PRIVATE ${MG_WARNINGS})

# ---- CLI ---------------------------------------------------------------------
add_executable(mg_cli Cli/main.cpp)
target_link_libraries(mg_cli PRIVATE mg_session)
target_compile_options(mg_cli PRIVATE ${MG_WARNINGS})
mg_copy_python(mg_cli)

# ---- Server: JSON API + the web editor in web/dist -----------------------------
find_package(httplib CONFIG REQUIRED)
add_executable(mg_server Server/main.cpp)
target_link_libraries(mg_server PRIVATE mg_session httplib::httplib)
target_compile_options(mg_server PRIVATE ${MG_WARNINGS})
mg_copy_python(mg_server)
