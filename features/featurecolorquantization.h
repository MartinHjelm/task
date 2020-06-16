#ifndef FEATURECOLORQUANTIZATION_H
#define FEATURECOLORQUANTIZATION_H

// std
#include <string>
#include <vector>

// PCL
#include "PCLTypedefs.h"

// OpenCV
#include <opencv2/core.hpp>

class FeatureColorQuantization
{
    bool printMsg;

    // Segmented point cloud representation of the object.
    pcl::PointCloud<PointT>::Ptr cloud_;
    // Index vector that contains the corresponding img coordinate for the object in the segmented img window.
    std::vector<std::vector<int> > pcPixelPos;
    // For each entry in the point cloud there is a idx value for the color chart
    std::vector<int> pcColorChartIdx;

    // This matrix contains the color chart indices for each of the pixels in the
    // original image.
    Eigen::MatrixXi colorCodesMat;


    // Image window of the segmented object
    cv::Mat img_;
    int offset_;

    // Image segmentation variables
    int N_comps; // Number of components found by the Fz-algorithm
    std::string rawfileName_;
    std::vector< std::vector<int> > cIndices; // Component indices that refer to the segmented image

    // Color quantization chart
    cv::Mat cqChart;
    std::vector<int> objColors;
    int N_colors;

    // Color segmentation functions
    void readQuantColorChart();
    void runGraphCutSegmentation();

public:
    std::vector<V3f> pcCQVals;
    std::vector<int> pcCQIdxs;

    FeatureColorQuantization();
    void setInputSource (const std::string &rawfileName, const cv::Mat &segmentedImg, const std::vector<std::vector<int> > &pcPxPos, int &offset, PC::Ptr &cloud);
    void getColorQuantizedImg(cv::Mat &cqImg);

    // Quantization functions
    void colorQuantizeImg();
    void colorQuantize();
    void imgCQ2PC();

    std::vector<double> computePointHist(const pcl::PointIndices::Ptr &points) const;
    std::vector<double> computePointHist2(const pcl::PointIndices::Ptr &points) const;
    std::vector<double> computeEntropyMeanVar(const pcl::PointIndices::Ptr &points) const;



};

#endif // FEATURECOLORQUANTIZATION_H
