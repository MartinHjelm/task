PROJECT( TC_Model )
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
FIND_PACKAGE(PCL 1.8 REQUIRED)

SET(MYINCS ${HOME}/Libs/include/helperlibs )
SET(MYLIBDIR ${HOME}/Libs/lib)
SET(MYLIBS ${MYLIBDIR}/libhelperlibs.a )

# Create a library
ADD_LIBRARY(TC_Model STATIC
  graspfactor.cpp
  graspsegmenter.cpp
  scenesegmentation.cpp
  symmetrycompletion.cpp
  graspsynthesis.cpp
  )


# Make sure the compiler can find include files for our library
TARGET_INCLUDE_DIRECTORIES (TC_Model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MYINCS} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
# Link dependencies
TARGET_LINK_LIBRARIES(TC_Model TC_Features ${GSL_LIBRARIES} ${MYLIBS} ${OpenCV_LIBS} ${PCL_LIBRARIES})
