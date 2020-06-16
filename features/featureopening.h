#ifndef FEATUREOPENING_H
#define FEATUREOPENING_H

// OpenCV
#include <opencv2/core.hpp>

// PCL
#include <PCLTypedefs.h>
#include <pcl/ModelCoefficients.h>
#include <mainaxes.h>

class FeatureOpening
{
  // Segmented out point cloud representation of the
  // object we are calculating the feature on.
  PC::Ptr cloud;
  PCN::Ptr cloudNormals;

  // Image window of the segmented object
  cv::Mat sIgm;

  // MODEL PARAMETERS
  // Circle: center point, radius and normal
  pcl::ModelCoefficients::Ptr pmCircle;
  pcl::PointIndices::Ptr circleInliers;
  V4f cCenter,cNormal;
  double radius;
  double maxRadius_;
  V3f mainAxis_,objcenter_;

  bool isCircleGood ();
  //void detect2DEllipses();
  void circle3DFit();
  float signedAngleBetweenVectors(const V3f& v1, const V3f& v2, const V3f& v3);
  bool isCircleOK(Eigen::VectorXf &model_coefficients, std::vector<int> &inliers);

public:
  FeatureOpening();
  void setInputSource(const PC::Ptr &cldPtr, const PCN::Ptr &cldNrmlPtr, const MainAxes &MA, const cv::Mat& imgWindow);

  // Detection functions
  bool hasOpening;
  void detectOpening();
  std::vector<double> computePosRelativeOpening ( const V3f &approachVec, const V3f &pos ) const;

  void generateCloudCircle ( PC::Ptr &cloud );
};

#endif // FEATUREOPENING_H
