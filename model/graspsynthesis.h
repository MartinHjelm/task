#ifndef GRASPSYNTHESIS_H
#define GRASPSYNTHESIS_H

// STD
#include <vector>

// Mine
#include "PCLTypedefs.h"
#include <cuboid.hpp>

class GraspSynthesis
{
    // Segmented out point cloud representation of the object we are computing grasps on
    PC::Ptr cloud_;
    PCN::Ptr cloudNormals_;

    // Heuristic checks for sampled grasps
    V4f plCf_;
    bool isSampledGraspOK(cuboid &graspBB );
    bool arePointsInsideBoundingBoxEvenlyDistributed (const PC::Ptr &cloud, const cuboid &bb);
//    std::vector<V4f> graspPlaneVec;


public:
    GraspSynthesis();
    void setInputSource(const PC::Ptr cloud, const PCN::Ptr cloudNormals, const V4f &plCf);

    bool sampleGrasp( cuboid &graspBB );
    bool sampleGraspFromCloud( cuboid &graspBB );
    bool sampleGraspCentroid( cuboid &graspBB );
    bool sampleGraspPoints( cuboid &graspBB );

};

#endif // GRASPSYNTHESIS_H
