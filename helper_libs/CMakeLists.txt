cmake_minimum_required(VERSION 3.5)
IF(APPLE)
  SET(HOME /Users/martinhjelm)
  SET(CMAKE_CXX_FLAGS "-std=c++11 -O4 -stdlib=libc++ -fPIC -Wno-unused-variable -DNDEBUG -march=native")
ELSE(APPLE)
  SET(HOME /home/martin)
  SET(CMAKE_CXX_FLAGS "-std=c++0x -O4 -fPIC -fopenmp -Wno-unused-variable  -DNDEBUG -march=native")
  SET (CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH} /home/martin/.linuxbrew/Cellar/pcl/1.8.0_6/share/)  
ENDIF(APPLE)

# Find PCL, OPENCV, BOOST, GLEW
FIND_PACKAGE(PCL 1.8 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS random system filesystem thread date_time iostreams serialization chrono REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)

ADD_DEFINITIONS(${PCL_DEFINITIONS})
INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	${PCL_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS})

ADD_LIBRARY(helperlibs myhelperfuns.cpp boosthelperfuns.cpp opencvhelperfuns.cpp pclhelperfuns.cpp eigenhelperfuns.cpp pca.cpp kmeans.cpp kmedoids.cpp bagofwords.cpp cuboid.cpp ${OPENNI_LIBRARY} )
TARGET_LINK_LIBRARIES(helperlibs  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES})


########################
SET(LIBDIR ${HOME}/Libs/lib)
SET(INCLUDEDIR ${HOME}/Libs/include/helperlibs)

 INSTALL(TARGETS helperlibs
         DESTINATION ${LIBDIR}
         )
 INSTALL(FILES myhelperfuns.h boosthelperfuns.h opencvhelperfuns.h pclhelperfuns.h eigenhelperfuns.h pca.h kmeans.h kmedoids.h bagofwords.h PCLTypedefs.h cuboid.hpp
         DESTINATION ${INCLUDEDIR}
         )
