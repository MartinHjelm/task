#ifndef GRASP_H
#define GRASP_H

// std
#include <vector>

// Mine
#include "PCLTypedefs.h"
#include "cuboid.hpp"
#include "scenesegmentation.h"
#include "mainaxes.h"
#include "featurecolorquantization.h"
#include "featureopening.h"
#include "featurecolorhist.h"
// #include "featurefpfh.h"
#include "ftr_fpfh_knn.h"
#include "featureposesurfaceangle.h"
#include "featureposesurfaceangle.h"
#include "featurehog.h"

/* Class for computing the features of a grasp given by a human demonstrator.
 * The class contains contains functions for segmenting out a human grap on an
 * object given a segmented scence of the object plus a scene in which the
 * human grasps the object.
 *
 * Given the segmented grasp it can then compute the local features for that grasp.
 *
*/

class GraspSegmenter
{
    void pushOnEigenXVec2StdVec(const Eigen::VectorXf &eigVec, std::vector<double> &stdVec);
    void detectPointCloudDiffByOrigID(const PC::Ptr &cloud1, const PC::Ptr &cloud2, pcl::PointIndices::Ptr &indices);
    void fitCuboid2PointCloud(const PC::Ptr &cloudOrig, const V3f &tableNrml, cuboid &graspBB);

public:

    GraspSegmenter();

    /*  gCloudSceneIdxs - Point indices for the grasped part of the object in the scene where the object is not grasped.
     *  gCloudSegmentIdxs  - Point indices for the grasped part of the object in the segmented scene where the object is not grasped.
     *  gImgIdxs - Image pixels indices for the grasped part of the object in the scene where the object is not grasped.
     *  graspBB - Bounding box for the grasp of the object. */
    pcl::PointIndices::Ptr gCloudSceneIdxs;
    pcl::PointIndices::Ptr gCloudSegmentIdxs;
    pcl::PointIndices::Ptr gCloudSegmentOrigIdxs;
    std::vector<std::vector<int> > gImgIdxs;
    cuboid graspBB;


    PC::Ptr pcGraspedObjectPart;

    void filterSkinColor(const cv::Mat &img,cv::Mat &mask);
    void filterCloudByPlane(const V4f &plCf, PC::Ptr &cloud);

    bool computeFeaturesFromGrasp(
    const SceneSegmentation &objSS,
    const MainAxes &FMA,
    const FeatureOpening &FO,
    const FeatureColorHist &FCH,
    const FeatureColorQuantization &FCQ,
    FeatureFPFHBoW &Ffpfh,
    featureHOG &fHOG,
    const FeaturePoseSurfaceAngle &FPSA,
    std::vector<double> &featureV);
    void computeGraspPoints(const SceneSegmentation &objSS, const SceneSegmentation &graspSS);
    void computeGraspPointsCube(const SceneSegmentation &objSS, const SceneSegmentation &graspSS);
    void computeGraspPointsSK (const SceneSegmentation &objSS, SceneSegmentation &graspSS);
    void computeGraspPointsGlove (const SceneSegmentation &objSS, SceneSegmentation &graspSS);
    void computeGraspPointsCube2 (const SceneSegmentation &objSS, const SceneSegmentation &graspSS);


};

#endif // GRASP_H
