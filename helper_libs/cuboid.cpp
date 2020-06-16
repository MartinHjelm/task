#include "cuboid.hpp"
#include "myhelperfuns.h"

cuboid::cuboid(): 
width(0),height(0),depth(0),
x_min(0),x_max(0),
y_min(0),y_max(0),
z_min(0),z_max(0),
axisMat(Eigen::MatrixXf::Identity(3,3)),
transVec(0,0,0),
axisRotMatSet(false){};

cuboid::cuboid( const cuboid& c ) :
width(c.width), height(c.height),
depth(c.depth),
x_min(c.x_min),x_max(c.x_max),
y_min(c.y_min),y_max(c.y_max),
z_min(c.z_min),z_max(c.z_max),
transVec(c.transVec),
axisMat(c.axisMat),
quartVec(c.quartVec),
axisRotMatSet(false) 
{};


void
cuboid::cpyCuboid(cuboid &c)
{
    c.width = width;
    c.height = height;
    c.depth = depth;
    c.transVec = transVec;
    c.quartVec = quartVec;
    c.quartVec.normalize();
    c.axisMat = axisMat;
    c.x_min = x_min;  c.x_max = x_max;
    c.y_min = y_min;  c.y_max = y_max;
    c.z_min = z_min;  c.z_max = z_max;    
    c.setAxisRotMat();
    c.setXYZminmax();
}

void
cuboid::setAxisRotMat()
{
  axisRotMat = axisMat;
  axisBackRotMat = axisRotMat.inverse();
}

V3f
cuboid::rotVecBack(const V3f &pt)
{
  return axisRotMat * pt + transVec;
}

void cuboid::setXYZminmax()
{

    // setAxisRotMat();

    // std::vector<double> xVals;
    // std::vector<double> yVals;
    // std::vector<double> zVals;


    // Eigen::MatrixXf cornerMat(8,3);
    // getCornersOfCuboid(cornerMat);
    // for(int i = 0; i!=8; i++)
    // {
    //   V3f v =  cornerMat.row(i);
    //   xVals.push_back(v(0));
    //   yVals.push_back(v(1));
    //   zVals.push_back(v(2));
    // }    

    // x_min = MyHelperFuns::minValVector(xVals);
    // y_min = MyHelperFuns::minValVector(yVals);
    // z_min = MyHelperFuns::minValVector(zVals);

    // x_max = MyHelperFuns::maxValVector(xVals);
    // y_max = MyHelperFuns::maxValVector(yVals);
    // z_max = MyHelperFuns::maxValVector(zVals);


    double x_min = -0.5f*width;
    double x_max = 0.5f*width;
    double y_min = -0.5f*height;
    double y_max = 0.5f*height;
    double z_min = -0.5f*depth;
    double z_max = 0.5f*depth;

}


void 
cuboid::setHeight(double h)
{
  height = h;
  y_min = -h/2;
  y_max = -y_min;
}


void 
cuboid::print()
{
  std::cout << " Width: " << width << " Height: " << height << " Depth: " << depth << std::endl;
  std::cout << "X min-max: " <<  x_min << " " << x_max << std::endl;
  std::cout << "Y min-max: " <<  y_min << " " << y_max << std::endl;
  std::cout << "Z min-max: " <<  z_min << " " << z_max << std::endl;
  std::cout << "Axis Matrix: \n" << axisMat << std::endl;
  std::cout << "Transvec: (" << transVec(0) << "," << transVec(1) << "," << transVec(2) << ")" << std::endl;
}



void
cuboid::getCornersOfCuboid( Eigen::MatrixXf &cornerMat ) const 
{

  assert(cornerMat.rows()==8);

  V3f n1(0.5 * width,0,0);
  V3f n2(0,0.5 * height,0);
  V3f n3(0,0,0.5 * depth);

  int matIdx = 0;
  for(int i = 0; i!=2; i++)
    for(int j = 0; j!=2; j++)
      for(int k = 0; k!=2; k++)
      {
        cornerMat.row(matIdx) = std::pow(-1,i) * n1
        + std::pow(-1,j) * n2 
        + std::pow(-1,k) * n3;
        matIdx++;
    }

  //(8x3 * 3*3)
    cornerMat = cornerMat * axisRotMat.transpose() ;
    cornerMat.rowwise() += transVec.transpose();
}


void
cuboid::getCornersOfCuboidAsPC( PC::Ptr &cloud ) const 
{
    Eigen::MatrixXf cornerMat(8,3);
    getCornersOfCuboid(cornerMat);
    
    for(int i=0; i!=8; i++)
    {
        PointT pt;
        pt.x = cornerMat(i,0);
        pt.y = cornerMat(i,1);
        pt.z = cornerMat(i,2);
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        cloud->push_back(pt);
    }
}



bool
cuboid::isPtInCuboid ( const V3f &pt ) const
{
  /* For each orthogonal direction of the cuboid compute the distance
   * to the center, that is, project each point onto the coordinate
   * system of the cuboid.
   */

    // Transform into cuboid coordinate system.
    V3f ptProj = axisBackRotMat * (pt-transVec);

    bool inX = (ptProj(0)>x_min-0 && ptProj(0)<x_max+0);
    bool inY = (ptProj(1)>y_min-0 && ptProj(1)<y_max+0);
    bool inZ = (ptProj(2)>z_min-0 && ptProj(2)<z_max+0);

    if( inX && inY && inZ )
        return true;
    else
        return false;
}



bool
cuboid::isPtInlier ( const V3f &pt, const float &dThreshold ) const
{
  /* For each orthogonal direction of the cuboid compute the distance
   * to the center, that is, project each point onto the coordinate
   * system of the cuboid.
   */

    // Transform into cuboid coordinate system.
    V3f ptProj = axisBackRotMat * (pt-transVec);

    bool inX = (std::fabs(ptProj(0)-x_min) < dThreshold || std::fabs(ptProj(0)-x_max) < dThreshold);
    bool inY = (std::fabs(ptProj(1)-y_min) < dThreshold || std::fabs(ptProj(1)-y_max) < dThreshold);
    bool inZ = (std::fabs(ptProj(2)-z_min) < dThreshold || std::fabs(ptProj(2)-z_max) < dThreshold);

    if( inX || inY || inZ )
        return true;
    else
        return false;
}



double
cuboid::shortestSideDist ( const V3f &pt ) const
{
    // Project onto cuboid axes.
    V3f ptProj = axisBackRotMat * (pt-transVec);

    double shortestDist = 1E10;
    double tmpDist = 0.0;

    // Shortest distance is always the orthogonal distance to one of the sides
  if( std::fabs (std::fabs(ptProj(0))-x_min ) < shortestDist )
      shortestDist = std::fabs (std::fabs(ptProj(0))-x_min );
  if( std::fabs (std::fabs(ptProj(0))-x_max ) < shortestDist )
      shortestDist = std::fabs (std::fabs(ptProj(0))-x_max );

  if( std::fabs (std::fabs(ptProj(1))-y_min ) < shortestDist )
      shortestDist = std::fabs (std::fabs(ptProj(1))-y_min );
  if( std::fabs (std::fabs(ptProj(1))-y_max ) < shortestDist )
      shortestDist = std::fabs (std::fabs(ptProj(1))-y_max );

  if( std::fabs (std::fabs(ptProj(2))-z_min ) < shortestDist )
      shortestDist = std::fabs (std::fabs(ptProj(2))-z_min );    
  if( std::fabs (std::fabs(ptProj(2))-z_max ) < shortestDist )
      shortestDist = std::fabs (std::fabs(ptProj(2))-z_max );    

    // printf("Shortest dist: %f\n",shortestDist);

  return shortestDist;
}


double
cuboid::shortestAngularDist ( const V3f &pt, const V3f &ptNormal ) const
{
    // V3f n1 = axisMat.col(0);
    // V3f n2 = axisMat.col(1);
    // V3f n3 = axisMat.col(2);

    // // Compute angular distance 
    // double angle1 = std::fabs (std::acos(ptNormal.dot(n1)/(ptNormal.norm()*n1.norm())) );
    // double angle2 = std::fabs (std::acos(ptNormal.dot(n2)/(ptNormal.norm()*n2.norm())) );
    // double angle3 = std::fabs (std::acos(ptNormal.dot(n3)/(ptNormal.norm()*n3.norm())) );
    // angle1 = (std::min) (angle1, M_PI - angle1);
    // angle2 = (std::min) (angle2, M_PI - angle2);
    // angle3 = (std::min) (angle3, M_PI - angle3);

    // if(angle1<angle2 && angle1<angle3)
    //     return angle1;
    // if(angle2<angle1 && angle2<angle3)
    //     return angle2;

    // return angle3;


    int side = 0;
    double shortestDist = 1E10;
    V3f ptProj = axisBackRotMat * (pt-transVec);


  if( std::fabs (std::fabs(ptProj(0))-x_min ) < shortestDist )
  {
      shortestDist = std::fabs (std::fabs(ptProj(0))-x_min );
      side = 0;
    }
  if( std::fabs (std::fabs(ptProj(0))-x_max ) < shortestDist )
  {
      shortestDist = std::fabs (std::fabs(ptProj(0))-x_max );
      side = 0;
  }
  if( std::fabs (std::fabs(ptProj(1))-y_min ) < shortestDist )
  {
      shortestDist = std::fabs (std::fabs(ptProj(1))-y_min );
      side = 1;
  }
  if( std::fabs (std::fabs(ptProj(1))-y_max ) < shortestDist )
  {
      shortestDist = std::fabs (std::fabs(ptProj(1))-y_max );
      side = 1;
  }
  if( std::fabs (std::fabs(ptProj(2))-z_min ) < shortestDist )
  {
      shortestDist = std::fabs (std::fabs(ptProj(2))-z_min );    
      side = 2;
      }
  if( std::fabs (std::fabs(ptProj(2))-z_max ) < shortestDist )
  {
      shortestDist = std::fabs (std::fabs(ptProj(2))-z_max );   
      side = 2;
      }

    // side = 2;
    // if(xd < yd && xd < zd)
    //     side = 0;
    // if(yd < xd && yd < zd)
    //     side = 1;

    V3f n = axisMat.col(side);
    double angle = std::fabs(std::acos(axisMat.col(side).normalized ().dot (ptNormal.normalized ()) ));
    // double angle = std::fabs(std::acos(ptNormal.dot(n)/(ptNormal.norm()*n.norm())));
    // printf("Angle: %f\n", RAD2DEG(angle));

    angle = (std::min) (angle, M_PI - angle);
    return angle;
}




