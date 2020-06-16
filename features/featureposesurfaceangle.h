#ifndef FEATUREPOSESURFACEANGLE_H
#define FEATUREPOSESURFACEANGLE_H

// PCL
#include <PCLTypedefs.h>

// Mine
#include "cuboid.hpp"


class FeaturePoseSurfaceAngle
{
    // POINT CLOUD
    // Segmented out point cloud representation of the
    // object we are calculating the feature on.
    PC::Ptr cloud;
    PCN::Ptr cloudNormals;

public:
    FeaturePoseSurfaceAngle();
    void setInputSource(const PC::Ptr &cldPtr, const PCN::Ptr &cldNrmlPtr);
    std::vector<double> compute(const cuboid &c, const pcl::PointIndices::Ptr &points) const;
    std::vector<double> compute(const V3f &approachVec, const pcl::PointIndices::Ptr &points) const;
};

#endif // FEATUREPOSESURFACEANGLE_H
