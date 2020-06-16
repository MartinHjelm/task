project( TC_Features )
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

IF(APPLE)
  SET(HOME /Users/martinhjelm)
  SET(CMAKE_CXX_FLAGS "-std=c++11 -O4 -stdlib=libc++ -Wno-unused-variable -Wnullability-completeness -Wno-deprecated-declarations -Wno-inconsistent-missing-override -march=native")
  SET(EXECUTABLE_FLAG MACOSX_BUNDLE)
ELSE(APPLE)
  SET(HOME /home/martin)
  SET(CMAKE_CXX_FLAGS "-std=c++11 -O4 -fPIC -fopenmp -Wno-unused-variable -Wall -Wextra -Wno-deprecated-declarations -Wno-unused-function -Wno-comment -Wno-unused-parameter -march=native")
  SET (CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH} /home/martin/.linuxbrew/Cellar/pcl/1.8.0_6/share/)
ENDIF(APPLE)


FIND_PACKAGE(GSL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED )


# Create a library
ADD_LIBRARY(TC_Features STATIC
  featurecolorhist.cpp
  featurecolorquantization.cpp
  featurefpfh.cpp
  ftr_fpfh_knn.cpp
  featureopening.cpp
  featureposesurfaceangle.cpp
  featuretexture.cpp
  featurehog.cpp
  mainaxes.cpp
  )

SET(VLFEAT_INCLUDE_DIRS "../vlfeat")
ADD_DEFINITIONS("-DVL_DISABLE_AVX")
FILE(GLOB SOURCES ../vlfeat/vl/*.c)
MESSAGE(STATUS "    sources path: ${SOURCES}")
ADD_LIBRARY(vlfeats ${SOURCES} )

# Make sure the compiler can find include files for our library
TARGET_INCLUDE_DIRECTORIES (TC_Features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )

# Link dependencies
TARGET_LINK_LIBRARIES(TC_Features TC_Ransac vlfeats ${MYLIBS} ${OpenCV_LIBS})
