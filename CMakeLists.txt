cmake_minimum_required(VERSION 3.12)
project(Lenet_5_CNN C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add your source files
set(SOURCES
    src/cnn.c
)

# Add your header files
set(HEADERS
    include/cnn.h
)

# Specify the include directories
include_directories(include)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} m)
