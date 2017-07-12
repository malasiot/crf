cmake_minimum_required(VERSION 2.8.0)

project(cvx_orec_crf)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake/ ${CMAKE_MODULE_PATH} )

find_package(OpenMP)

if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(PCL REQUIRED)
find_package(Boost 1.49 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    cvx_util
    cvx_renderer
)

catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES
        cvx_orec_crf
)

add_definitions(-std=c++11)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

IF ( CMAKE_COMPILER_IS_GNUCXX )
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wreturn-type" )
ENDIF( CMAKE_COMPILER_IS_GNUCXX )


SET(CERTH_OBJECT_DETECTOR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/cvx/orec/)

# add subprojects
ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(test)

