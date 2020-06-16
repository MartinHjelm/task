#ifndef SAC3DCIRCLEWITHNORMAL_H
#define SAC3DCIRCLEWITHNORMAL_H

#include "PCLTypedefs.h"

#include <vector>

class SAC3DCircleWithNormal
{
public:


  // Cloud
  PC::Ptr cloud;
  PCN::Ptr cloudNormals;
  pcl::PointIndices::Ptr indices_;
  pcl::PointIndices::Ptr inliers;

  // Model settings
  float radius_min_;
  float radius_max_;

  SAC3DCircleWithNormal(PC::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr cloudNormals, pcl::PointIndices::Ptr inliers);

  bool isSampleGood(const std::vector<int> &samples, const V3f &axis);
  bool isSampleGood2 (Eigen::VectorXf &model_coefficients, const V3f &axis, const V3f &center, const double &maxRadius );
  bool computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients);

  void
  getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances);


  void
  selectWithinDistance (
      const Eigen::VectorXf &model_coefficients, const double threshold,
      std::vector<int> &inliers);


  int
  countWithinDistance (
      const Eigen::VectorXf &model_coefficients, const double threshold);



//  void
//  optimizeModelCoefficients (
//        const std::vector<int> &inliers,
//        const Eigen::VectorXf &model_coefficients,
//        Eigen::VectorXf &optimized_coefficients);



//  void
//  projectPoints (const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients,
//        PC &projected_points, bool copy_data_fields);


  bool
  doSamplesVerifyModel (
        const std::set<int> &indices,
        const Eigen::VectorXf &model_coefficients,
        const double threshold);

  bool
  isModelValid (const Eigen::VectorXf &model_coefficients);

};

#endif // SAC3DCIRCLEWITHNORMAL_H
