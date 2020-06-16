#ifndef PCLHELPERFUNS_H
#define PCLHELPERFUNS_H

// std
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

// Mine
#include <PCLTypedefs.h>
#include <cuboid.hpp>

// OpenCV
#include <opencv2/core/core.hpp>

class PCLHelperFuns
{
public:
  PCLHelperFuns();

  // PCL
  static int readPCDFile(const std::string &fileName, PC::Ptr &cloud);
  static void colorIndices(PC::Ptr &cloud, const pcl::PointIndices::Ptr &indPtr, const V3f &rgb);
  static void convEigen2PCL(const V3f &eigVec, PointT &pt);
  static int detectPointCloudDiff(const PC::Ptr &cloud1, const PC::Ptr &cloud2, pcl::PointIndices::Ptr &indices);
  static void detectPointCloudDiffByOrigID(const PC::Ptr &cloud1, const PC::Ptr &cloud2, pcl::PointIndices::Ptr &indices);
  static void detectImageDiff(cv::Mat &img1, cv::Mat &img2, pcl::PointIndices::Ptr &ptsIdx, std::vector<std::vector<int> > &indices, const bool diffOrSim = true, const int &pxDist=20 );
  static void filterByOrigID(const PC::Ptr &cloudIn, const pcl::PointIndices::Ptr &ptIdxs, PC::Ptr &cloudOut);
  static void filterOutByOrigID(const pcl::PointIndices::Ptr &ptOrigIdxs, const PC::Ptr &cloudIn, PC::Ptr &cloudOut);
  static void filterOutSmallClusters(PC::Ptr &cloud, const float &clusterTolerance=0.01, const int &minClusterSize=50);
  static void fitCuboid2PointCloudPlane(const PC::Ptr &cloudOrig, const V3f &tableNrml, cuboid &graspBB);

  static void filterCloudFromCloud(const PC::Ptr &cloud2Filter,const PC::Ptr &cloud2FilterOut, PC::Ptr &cloudOut,float voxSize=0.01);


  static void unpackPointColor(const uint32_t &rgba, std::vector<int> &rgbaVec);

  static void matchIndices2Scene( const PC::Ptr &cluster, const PC::Ptr &cloud, pcl::PointIndices::Ptr &ptIdxs);
  static void matchIndices2SceneByOrigID( const PC::Ptr &cluster, const PC::Ptr &cloud, pcl::PointIndices::Ptr &ptIdxs);
  static V3f ptAtorigID(const PC::Ptr &cloud, const int &ID );
  static int ID2idx(const PC::Ptr &cloud, const int &ID );

  static void ID2Img(const pcl::PointIndices::Ptr &ptIdxs, const cv::Mat &ImgOrig, cv::Mat &imOut);
  static std::pair<int,int> arIdx2MatPos(const int &idx, const int &imWidth);


  static bool isPxInCloud(const PC::Ptr &cloud, const int &pxID, V3f &pt);

  static void getOrigInd(const PC::Ptr &cloud, const pcl::PointIndices::Ptr &ptIdxsIn, pcl::PointIndices::Ptr &ptIdxsOut );

  static int findCloudOffset( const PC::Ptr &cluster, const PC::Ptr &cloud );
  static void mergePointIndices(pcl::PointIndices::Ptr &pis1, const pcl::PointIndices::Ptr &pis2 );

  static void selectPointsInCube( const PC::Ptr cloud, const cuboid &cube, pcl::PointIndices::Ptr indPtr );
  static void computePointCloudBoundingBox(const PC::Ptr &cloud, cuboid &cube);
  static void computePointsInsideBoundingBox (const PC::Ptr &cloud, const cuboid &bb, pcl::PointIndices::Ptr &pIdxs);
  static bool arePointsInsideBoundingBoxEvenlyDistributed (const PC::Ptr &cloud, const cuboid &bb);


  static int computeInliersToCube(const PC::Ptr &cloud, const cuboid &bb, const double &dThreshold);

  static void pt2px(const int &cols, const int &id, std::vector<int> &pxPos );
  static void pt2px(const int &cols, const int &id, int &row, int &col);
  static void pt2px(const int &cols, const PointT &pt, int &row, int &col);
  static void projectPt2Img(const PointT &pt, int &row, int &col);
  static void cloud2img(const PC::Ptr &cloud, const bool &hasColor, cv::Mat &img);


  static void cloud2origInd(const PC::Ptr &cloud, pcl::PointIndices::Ptr &ptIdxs );
  static void smoothCloud(PC::Ptr &cloud);
  static void smoothCloud2(const PC::Ptr &cloudIn, PC::Ptr &cloudOut);
  static void smoothCloud3(const cv::Mat &im, PC::Ptr &cloud, PCN::Ptr &cloudNormals);
  static void smoothCloud3(PC::Ptr &cloud, PCN::Ptr &cloudNormals);
  static void computeCloudNormals(const PC::Ptr &cloud, const float &normal_radius, PCN::Ptr &normals);
  static void samplePlane( const PC::Ptr &cloud, V4f &planeParams, V3f &planePt );
  static inline int randIdx(const int &Nmax){ return std::rand() % Nmax; };
};


#endif // PCLHELPERFUNS_H
