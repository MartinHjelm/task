#ifndef GRASPFACTOR_H
#define GRASPFACTOR_H

// Mine
#include "PCLTypedefs.h"
#include "cuboid.hpp"
#include "scenesegmentation.h"
#include "mainaxes.h"
#include "featurecolorquantization.h"
#include "featurecolorhist.h"
#include "featureopening.h"
// #include "featurefpfh.h"
#include "ftr_fpfh_knn.h"
#include "graspsegmenter.h"
#include "featureposesurfaceangle.h"

/* Class that given a scene with an object and approach vector computes the
 * features from the bounding box of the gripper.
 */
class GraspFactor
{
    SceneSegmentation SS_;
    /* Features */
    MainAxes FMA;
    FeatureOpening FO;
    FeatureColorQuantization FCQ;
    FeatureColorHist FCH;
    // featureFPFH Ffpfh;
    FeatureFPFHBoW Ffpfh;
    FeaturePoseSurfaceAngle FPSA;

    bool featuresComputed;


    // LMNN and KNN containers
    Eigen::MatrixXf xpPoints; // Dorky names...
    Eigen::MatrixXf distMat;
    Eigen::VectorXf metricVec;
    std::vector<int> labels;

    // Feature variables and functions
    static int featureLen;
    Eigen::MatrixXf meanXP, covXP, centered, stdXP;

    void scaleFeatureVector(Eigen::MatrixXf &mat);
    void scaleFeatureVector(Eigen::VectorXf &vec);
    void scaleFeatureVectorVariance(Eigen::MatrixXf &mat);
    void scaleFeatureVectorVariance(Eigen::VectorXf &vec);


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Stores all generated grasps
    std::vector<cuboid> graspCuboidsVec;
    std::vector<double> graspFactorVec;

    GraspFactor(SceneSegmentation &SS);

    void loadData(const std::string &taskName);
    void computeFeaturesOverObject();
    void computeFeaturesFromGrasp(const pcl::PointIndices::Ptr &graspedPtsIdxs, std::vector<double> &featureVec, const cuboid &graspBB);
    bool computeGraspProbability(const cuboid &graspBB, const int kN, double &nll);


    void runGraspPlanner(const int &Nsamples, const int &kN=3, const int &cldOrPlane=0);
    inline void wait_on_enter()
    {
        std::string dummy;
        std::cout << "Enter to continue..." << std::endl;
        std::getline(std::cin, dummy);
    }

};


#endif // GRASPFACTOR_H
