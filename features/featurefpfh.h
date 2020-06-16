#ifndef FEATUREFPFH_H
#define FEATUREFPFH_H

// DEFS
#include "PCLTypedefs.h"

// MINE

class featureFPFH
{
    // Cloud
    PC::Ptr cloud_;
    PCN::Ptr cloudNormals_;

    // FPFH descriptor
    double fpfhRadiusSearch_;
    double normalRadiusSearch_;
    Eigen::MatrixXd featureMat_;

    // Feature encoding
    Eigen::MatrixXd gmmMeans_;
    Eigen::MatrixXd gmmCovs_;
    Eigen::VectorXd gmmPriors_;
    Eigen::MatrixXd pcaProjMat_;
    int codeBookSize_;
    int fpfhProjDim_;
    std::vector<int> objBoWCode_;

    void computeFPFHfeature();
    void encodeFeature();

public:
    // Constructor
    featureFPFH();
    void setInputSource(const PC::Ptr &cPtr,const PCN::Ptr &cnPtr);
    void setCodeBook(const std::string &normalRadius, const std::string &fpfhRadius, const std::string &codeBookSize, const std::string &fpfhProjDim_);

    // Computes the FPFH Bag-of-Words representation from the given point cloud.
    void cptBoWRepresentation();
    // Returns the specific Bag-Of-Words codes for specific point cloud points
    void getPtsBoWHist(const pcl::PointIndices::Ptr &points, std::vector<double> &cwhist) const;
    // Returns the specific Bag-Of-Words codes for all points in the cloud
    void getObjBoWHist(std::vector<double> &cwhist);
    int getBoWForPoint(uint idx);
};

#endif // FEATUREFPFH_H
