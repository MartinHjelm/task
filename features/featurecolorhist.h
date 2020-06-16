#ifndef FEATURECOLORHIST_H
#define FEATURECOLORHIST_H

// STD
#include <vector>

// MINE
#include "PCLTypedefs.h"

// Open CV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


/* Class of features that output various histograms of a given point cloud.*/

class FeatureColorHist
{

  cv::Mat img_; // Img of complete scene
  // Img of complete scene applied Sobel filter for different gradients
  cv::Mat imgGrad1_;
  cv::Mat imgGrad2_;
  cv::Mat imgGrad3_;
  cv::Mat brightnessMat_; // Brightness component of HSV color space of complete scene

  int bins_;

  // Converts a one channel mat to a std vector.
  std::vector<double> mat2Vec(const cv::Mat &mat) const;
  // Converts between array and matrix position
  std::pair<int,int> arIdx2MatPos(const int &idx, const int &imWidth=640) const;

  // Filter functions
  void computeImgGrad(const cv::Mat &imgIn, cv::Mat &imgOut, const int &gradOrder);
  void computeBrightness(const cv::Mat &imgIn, cv::Mat &imgOut);

  // Histogram functions
  void computeGenericHist(const cv::Mat &imgIn, const pcl::PointIndices::Ptr &points, const int &Nbins, std::vector<double> &hist ) const;
  void getZeroCrossings(float threshold, cv::Mat &binary, const cv::Mat &laplace);



public:
  FeatureColorHist();
  void setInputSource(const cv::Mat &img);
  void computeFeatureMats(); // Applies all filters

  // Computes histograms over filtered images for a set of points.
  void histGrad(const pcl::PointIndices::Ptr &points, const int &gradOrder, std::vector<double> &hist) const;
  void histBrightness(const pcl::PointIndices::Ptr &points, std::vector<double> &hist) const;

  double gradAtPoint(const int &idx, const int &gradOrder,const int &imwidth=640) const;
  int binAtPoint(const int &idx, const int &gradOrder, const int &imwidth=640) const;
};

#endif // FEATURECOLORHIST_H
