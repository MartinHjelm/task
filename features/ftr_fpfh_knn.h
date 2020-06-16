#ifndef FEATUREFPFH_H
#define FEATUREFPFH_H

// DEFS
#include "PCLTypedefs.h"

// MINE

class FeatureFPFHBoW
{
    // Cloud
    PC::Ptr cloud_;
    PCN::Ptr cloudNormals_;

    // FPFH descriptor
    double fpfhRadiusSearch_;
    double normalRadiusSearch_;
    Eigen::MatrixXd featureMat_;

    // Feature encoding
    Eigen::MatrixXd pcaProjMat_;
    Eigen::MatrixXd codeBook_;
    int codeBookSize_;
    int fpfhProjDim_;
    std::vector<int> objBoWCode_;

    void ComputeFPFHFeatures();
    void EncodeFeatures();

public:
    // Constructor
    FeatureFPFHBoW();
    void SetInputSource(const PC::Ptr &cPtr,const PCN::Ptr &cnPtr);
    void SetCodeBook(const std::string &normalRadius, const std::string &fpfhRadius, const std::string &codeBookSize, const std::string &fpfhProjDim_);

    // Computes the FPFH Bag-of-Words representation from the given point cloud.
    void CptBoWRepresentation();
    // Returns the specific Bag-Of-Words codes for specific point cloud points
    void GetPtsBoWHist(const pcl::PointIndices::Ptr &points, std::vector<double> &cwhist) const;
    // Returns the specific Bag-Of-Words codes for all points in the cloud
    void GetObjBoWHist(std::vector<double> &cwhist);
    int GetBoWForPoint(uint idx);
};

#endif // FEATUREFPFH_H
