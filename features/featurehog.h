#ifndef FEATUREHOG_H
#define FEATUREHOG_H

// STD
#include <vector>

// MINE
#include "PCLTypedefs.h"
#include "bagofwords.h"

// Open CV
#include "opencv2/core.hpp"


class featureHOG
{
    cv::Mat img_; // Segmented img of objects
    Eigen::MatrixXd hogMat_; // HoG Features Matrix
    std::vector<double> hogHist_;
    int codeBookSize_;
    BagOfWords bow;
    void computeHOG (cv::Mat &img, Eigen::MatrixXd &hogMat);

public:
    featureHOG();
    void setInputSource(const cv::Mat &img);
    void compute();
    std::vector<double> getFeature();
    void appendFeature(std::vector<double> &toVec);
};

#endif // FEATUREHOG_H
