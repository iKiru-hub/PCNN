cmake_minimum_required(VERSION 3.15)
project(pclib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add pybind11
add_subdirectory(extern/pybind11)

# Header files
set(HEADERS
    include/pcnn.hpp
    # include/utils.hpp
)

# Python module
set(MODULE_SOURCES
    src/binding.cpp
)

# Add the pybind11 module
pybind11_add_module(pclib ${MODULE_SOURCES})

# Test executable
set(TEST_SOURCES
    src/test.cpp
)

# Add the test executable
add_executable(pcnn_test ${TEST_SOURCES})

# Include directories for both targets
target_include_directories(pclib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(pcnn_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries for both targets
target_link_libraries(pclib
    PRIVATE
        Eigen3::Eigen
)

target_link_libraries(pcnn_test
    PRIVATE
        Eigen3::Eigen
)

# Set compiler flags for macOS
if(APPLE)
    # Flags for Python module
    target_compile_options(pclib PRIVATE 
        -stdlib=libc++ 
        -O3
        -Wall
        -Wextra
        -mmacosx-version-min=10.9
    )
    set_target_properties(pclib PROPERTIES
        LINK_FLAGS "-stdlib=libc++ -mmacosx-version-min=10.9"
    )

    # Flags for test executable
    target_compile_options(pcnn_test PRIVATE 
        -stdlib=libc++ 
        -O3
        -Wall
        -Wextra
        -mmacosx-version-min=10.9
    )
    set_target_properties(pcnn_test PROPERTIES
        LINK_FLAGS "-stdlib=libc++ -mmacosx-version-min=10.9"
    )
endif()


# suppress warnings
add_definitions(-w)

# print that the warning suppression is on
message(">>> Hello! Beware that warning suppression is on <<<")

