cmake_minimum_required(VERSION 2.8.4)

# Project name is the name of the directory
get_filename_component(ProjectId ${CMAKE_SOURCE_DIR} NAME)
string(REPLACE " " "_" ProjectId ${ProjectId})
project(${ProjectId})

# sets executable directory for generic, debug and release builds
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin-Debug" )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin-Release" )
add_executable(${ProjectId} Source.cpp)

# Linking OpenGL, GLUT packages
find_package(GLUT)
find_package(OpenGL)
INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${ProjectId} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

# defining Debug and Release build options
if (UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "-pedantic -Wall -Wextra -DDEBUG -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-pedantic -Wall -Wextra -g0 -o3")
endif ()

# debug build command: g++ Source.cpp -pedantic -Wall -Wextra -DDEBUG -g -lglut -lGLU -lGL -o grafika
# release build command: g++ Source.cpp -pedantic -Wall -Wextra -g0 -03 -lglut -lGLU -lGL -o grafika