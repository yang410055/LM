
cmake_minimum_required( VERSION 3.2 )

project(myLM)

set(SRCLIST main.cpp)

find_package(OpenCV REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( myLM ${SRCLIST} )

target_link_libraries( myLM ${OpenCV_LIBRARIES} ) 

