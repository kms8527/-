# %Tag(FULLTEXT)%
cmake_minimum_required(VERSION 2.8.3)
project(beginner_tutorials)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

## Declare a catkin package
catkin_package(
CATKIN_DEPENDS std_msgs roscpp
)
include_directories(${catkin_INCLUDE_DIRS})


add_executable(
node_1 src/A_publish.cpp
)

add_executable(
node_2 src/B_publish.cpp
)

add_executable(
node_3 src/add.cpp
)

add_executable(
node_4 src/average.cpp
)

target_link_libraries(node_1 ${catkin_LIBRARIES})
add_dependencies(node_1 ${catkin_LIBRARIES})

target_link_libraries(node_2 ${catkin_LIBRARIES})
add_dependencies(node_2 ${catkin_LIBRARIES})

target_link_libraries(node_3 ${catkin_LIBRARIES})
add_dependencies(node_3 ${catkin_LIBRARIES})

target_link_libraries(node_4 ${catkin_LIBRARIES})
add_dependencies(node_4 ${catkin_LIBRARIES})


