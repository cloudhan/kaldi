if(MKL_cmake_included)
    return()
endif()
set(MKL_cmake_included true)

function(detect_mkl LIBNAME)
    find_path(MKLINC mkl_cblas.h
        HINTS ${MKLROOT}/include $ENV{MKLROOT}/include)
    get_filename_component(__mkl_root "${MKLINC}" PATH)
    find_library(MKLLIB NAMES ${LIBNAME}
        PATHS ${__mkl_root}/lib ${__mkl_root}/lib/intel64
        NO_DEFAULT_PATH)

    if(WIN32)
        set(MKLREDIST ${__mkl_root}/../redist/)
        find_file(MKLDLL NAMES ${LIBNAME}.dll
            HINTS ${MKLREDIST}/mkl ${MKLREDIST}/intel64/mkl)
        if(NOT MKLDLL)
            return()
        endif()
    endif()

    if(WIN32)
        # Add paths to DLL to %PATH% on Windows
        get_filename_component(MKLDLLPATH "${MKLDLL}" PATH)
    endif()

    set(HAVE_MKL TRUE PARENT_SCOPE)
    set(MKLINC "${MKLINC}" PARENT_SCOPE)
    set(MKLLIB "${MKLLIB}" PARENT_SCOPE)
    set(MKLDLL "${MKLDLL}" PARENT_SCOPE)
endfunction()

detect_mkl("mkl_rt")

if(HAVE_MKL)
    set(MSG "Intel(R) MKL:")
    message(STATUS "${MSG} include ${MKLINC}")
    message(STATUS "${MSG} lib ${MKLLIB}")

    add_definitions(-DHAVE_MKL=1)
    include_directories(${MKLINC})
    link_libraries(${MKLLIB})
else()
    message(FATAL_ERROR "MKL is requested, but I cannot find it")
endif()
