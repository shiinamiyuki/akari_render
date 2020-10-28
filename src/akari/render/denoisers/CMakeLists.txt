if(AKR_ENABLE_OIDN)
    add_plugin(OIDNDenoiser oidn.cpp)
    target_include_directories(OIDNDenoiser PUBLIC ${OIDN_INCLUDE_DIRS})
    target_link_libraries(OIDNDenoiser OpenImageDenoise)
    add_custom_command(TARGET OIDNDenoiser POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:OpenImageDenoise>  ${CMAKE_BINARY_DIR}/dist/)
endif()