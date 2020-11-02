if(AKR_ENABLE_OIDN)
    add_plugin(OIDNDenoiser oidn.cpp)
    target_include_directories(OIDNDenoiser PUBLIC ${OIDN_INCLUDE_DIRS})
    target_link_libraries(OIDNDenoiser OpenImageDenoise)
    copy_to_dist(OIDNDenoiser)
endif()

if(AKR_ENABLE_NETWORK)
    add_plugin(NetworkWorld network-node.cpp)
    target_include_directories(NetworkWorld PUBLIC ${CMAKE_SOURCE_DIR}/external/asio/asio/include)
    copy_to_dist(NetworkWorld)
endif()

