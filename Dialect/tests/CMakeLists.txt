configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TANGENT_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        tangent-opt
        )

add_lit_testsuite(check-tangent "Running the Tangent regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${HELLO_TEST_DEPENDS}
        )
set_target_properties(check-tangent PROPERTIES FOLDER "Tests")

add_lit_testsuites(HELLO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${HELLO_TEST_DEPENDS})
