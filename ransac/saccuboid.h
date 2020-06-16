#ifndef SACCUBOID_H
#define SACCUBOID_H

#include "PCLTypedefs.h"
#include "cuboid.hpp"

class SacCuboid
{

    PC::Ptr cloud;
    PCN::Ptr cloudNormals;
    cuboid c_;

    // Cuboid plane parameters: ax+by+cz+d=0
    V4f pl1Params, pl2Params, pl3Params;
    //  int p1Idx, p2Idx, p3Idx;
    int p1Idx, p2Idx, p3Idx, p4Idx;
    int p1IdxBest, p2IdxBest, p3IdxBest;
    // Max angle diff from orthogonal for the planes
    double maxDegWiggle;

    // Rectangles of each plane
    std::vector<V3f> r1, r2, r3;

    // SAC PARAMS
    double distanceThreshold, normalDistanceWeight;
    uint maxIter;

    int computeEnclosingRectangle(const V4f &planeParams, std::vector<V3f>& rPoly);
    int computeModel(Eigen::VectorXf &modelCoefficients);

    double pointToPlaneDistance (const PointT &p, const V4f &plParams);
    double pointNormalToPlaneNormalDistance (const PointN &p, const V4f &plane_coefficients);
    // Checks that the cuboid planes are somewhat orthogonal
    bool isModelOK(const Eigen::VectorXf &modelCoefficients);


public:

    SacCuboid();

    // Setters
    void setInputCloud(const PC::Ptr &cPtr);
    void setInputNormals(const PCN::Ptr &cnPtr);
    void setDistanceThreshold(double d);
    void setNormalDistanceWeight(double dw);
    void setMaxIterations(uint n);

    // Fitting
    void sacFitCuboid2PointCloud(Eigen::VectorXf &modelCoefficients );
    void computeCuboidParams(const Eigen::VectorXf &model_coefficients);
    inline cuboid getModelParameters(){ return c_; }

    // Count and select inliers to model
    int countWithinDistance(const Eigen::VectorXf &modelCoefficients);
    int countWithinDistance2(const V4f &pl1,const V4f &pl2,const V4f &pl3);
    int countWithinDistance3(const cuboid &c);
    void selectWithinDistance(const Eigen::VectorXf &model_coefficients, pcl::PointIndices::Ptr &inliers);
    void selectWithinDistance2(pcl::PointIndices::Ptr &inliers);
    void selectWithinDistance3(const cuboid &c, pcl::PointIndices::Ptr &inliers);
    void selectWithinDistancePlane(const Eigen::VectorXf &modelCoefficients, std::vector<int> &inliers);

};

#endif // SACCUBOID_H
