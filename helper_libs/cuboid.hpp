#ifndef CUBOID_HPP
#define CUBOID_HPP

#include "PCLTypedefs.h"

class cuboid
{

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    double width,height,depth; // Order in relation to axes, and in the cuboid coordinate system
    V3f transVec;
    Eigen::Quaternionf quartVec;
    Eigen::Matrix3f axisMat; // Each col is an axis vector x,y,z
    float x_min, x_max, y_min, y_max, z_min, z_max; // In viewpoint coordinate system
    Eigen::Matrix3f axisRotMat; // For transforming vector into cube coordinate system
    Eigen::Matrix3f axisBackRotMat; // For transforming vector into original coordinate system
    bool axisRotMatSet;
    
    cuboid();
    cuboid( const cuboid& c );
    
    void setHeight(double h);
    void setXYZminmax();
    V3f rotVecBack(const V3f &pt);
    void setAxisRotMat();
    bool isPtInCuboid ( const V3f &pt ) const;
    bool isPtInlier ( const V3f &pt, const float &dThreshold ) const;
    double shortestSideDist ( const V3f &pt ) const;
    double shortestAngularDist ( const V3f &pt, const V3f &ptNormal ) const;
    void cpyCuboid( cuboid &c );
    void print();

    void getCornersOfCuboid( Eigen::MatrixXf &cornerMat ) const;
    void getCornersOfCuboidAsPC( PC::Ptr &cloud ) const;
};

#endif // CUBOID_HPP
