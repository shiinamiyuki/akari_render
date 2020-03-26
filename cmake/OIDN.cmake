option(AKARI_USE_OIDN "USE Intel OpenImageDenoise" OFF)

if(AKARI_USE_OIDN)
    include_directories(external/oidn/include)
    add_subdirectory(external/oidn)

endif()