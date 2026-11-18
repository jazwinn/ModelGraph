find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(mg_tests
    expr_test.cpp
    document_test.cpp
    analyze_test.cpp
    equivalence_test.cpp
    script_test.cpp
)
target_link_libraries(mg_tests PRIVATE
    mg_session
    InferenceEngine::Models
    GTest::gtest
    GTest::gtest_main
)
target_compile_options(mg_tests PRIVATE ${MG_WARNINGS})
target_compile_definitions(mg_tests PRIVATE MG_TEST_TMP="${CMAKE_CURRENT_BINARY_DIR}/tmp")
mg_copy_python(mg_tests)
gtest_discover_tests(mg_tests DISCOVERY_TIMEOUT 60)
