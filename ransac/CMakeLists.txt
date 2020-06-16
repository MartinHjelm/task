project( TC_Ransac ) 

# Create a library 
ADD_LIBRARY(TC_Ransac STATIC
  sac3dcirclewithnormal.cpp
  saccuboid.cpp
  )

# Make sure the compiler can find include files for our library
TARGET_INCLUDE_DIRECTORIES(TC_Ransac PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
TARGET_LINK_LIBRARIES(TC_Ransac ${MYLIBS})
