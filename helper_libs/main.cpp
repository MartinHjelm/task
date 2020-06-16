#include "PCLTypedefs.h"

// std  aq
#include <string>
#include <stdexcept>

// Mine
#include "cuboid.hpp"
#include "myhelperfuns.h"


bool test_cuboidFuns();

int
main(int argc, char **argv)
{
  if(test_cuboidFuns())
    std::cout << "Test done. Success." << std::endl;
  else
    std::cout << "Test failed." << std::endl;

  return 0;
}


bool test_cuboidFuns()
{
  cuboid cube;

  cube.x_min = -0.1; cube.x_max = 0.1;
  cube.y_min = -0.1; cube.y_max = 0.1;
  cube.z_min = -0.1; cube.z_max = 0.1;
  cube.height = cube.x_max - cube.x_min;
  cube.width = cube.y_max - cube.y_min;
  cube.depth = cube.z_max - cube.z_min;

  cube.transVec(0) = 0;
  cube.transVec(1) = 0;
  cube.transVec(2) = 0;
  cube.quartVec.w() =  1;
  cube.quartVec.x() =  0;
  cube.quartVec.y() =  0;
  cube.quartVec.z() =  0;
  cube.quartVec.normalize();
  cube.axisVec = Eigen::Matrix3f::Identity(3,3);
  cube.axisVec.row(0) = cube.quartVec._transformVector(cube.axisVec.row(0));
  cube.axisVec.row(1) = cube.quartVec._transformVector(cube.axisVec.row(1));
  cube.axisVec.row(2) = cube.quartVec._transformVector(cube.axisVec.row(2));

  MyHelperFuns::printEigenVec(cube.transVec,"Translation vector");
  MyHelperFuns::printEigenVec(cube.axisVec.row(0),"Axis 1");
  MyHelperFuns::printEigenVec(cube.axisVec.row(1),"Axis 2");
  MyHelperFuns::printEigenVec(cube.axisVec.row(2),"Axis 3");


  V3f pt(0.0999,0.05,0.05);
  return cube.isPtInCuboid(pt);
}
