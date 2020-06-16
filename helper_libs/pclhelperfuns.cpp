#include <pclhelperfuns.h>

// std
#include <fstream>
#include <iostream>
//#include <ctime>
//#include <random>

// Boost
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// PCL
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>
#include <pcl/octree/octree_pointcloud.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

PCLHelperFuns::PCLHelperFuns(){
};

void PCLHelperFuns::unpackPointColor(const uint32_t &rgba,
                                     std::vector<int> &rgbaVec) {
        rgbaVec.clear();
        rgbaVec.push_back((rgba >> 16) & 0x0000ff);
        rgbaVec.push_back((rgba >> 8) & 0x0000ff);
        rgbaVec.push_back((rgba)&0x0000ff);
}

/* Takes the original ID's from the point cloud and puts them into a point
 * indices  vector. */
void PCLHelperFuns::cloud2origInd(const PC::Ptr &cloud,
                                  pcl::PointIndices::Ptr &ptIdxs) {
        for (PC::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter)
                ptIdxs->indices.push_back(iter->ID);
}

bool PCLHelperFuns::isPxInCloud(const PC::Ptr &cloud, const int &pxID,
                                V3f &pt) {
        for (PC::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter)
                if (iter->ID == pxID) {
                        pt = iter->getVector3fMap();
                        return true;
                }
        return false;
}

void PCLHelperFuns::getOrigInd(const PC::Ptr &cloud,
                               const pcl::PointIndices::Ptr &ptIdxsIn,
                               pcl::PointIndices::Ptr &ptIdxsOut) {
        std::vector<int>::iterator iter = ptIdxsIn->indices.begin();
        for (; iter != ptIdxsIn->indices.end(); ++iter)
                ptIdxsOut->indices.push_back(cloud->points[*iter].ID);
}

V3f PCLHelperFuns::ptAtorigID(const PC::Ptr &cloud, const int &ID) {
        V3f pt(0.0, 0.0, 0.0);
        for (PC::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter) {
                if (iter->ID == ID) {
                        pt = iter->getVector3fMap();
                }
        }
        return pt;
}

int PCLHelperFuns::ID2idx(const PC::Ptr &cloud, const int &ID) {
        int counter = 0;
        for (PC::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter) {
                if (iter->ID == ID) {
                        return counter;
                }
                counter++;
        }
        return counter;
}

void PCLHelperFuns::pt2px(const int &cols, const int &id, int &row, int &col) {
        row = id / cols;
        col = id % cols;
}

/* Converts element position to matrix position. */
void PCLHelperFuns::pt2px(const int &cols, const int &id,
                          std::vector<int> &pxPos) {
        pxPos[0] = id / cols;
        pxPos[1] = id % cols;
}

void PCLHelperFuns::pt2px(const int &cols, const PointT &pt, int &row,
                          int &col) {
        row = pt.ID / cols;
        col = pt.ID % cols;
}

/* Works for kinect. */
void PCLHelperFuns::projectPt2Img(const PointT &pt, int &row, int &col) {
        double focalInv = 1000.0 / 525.0;
        int imageCenterX = 640 / 2;
        int imageCenterY = 480 / 2;

        // Compute approximate coordinates of pixel position of segmented object
        row = (1000.0 * pt.y / (pt.z * focalInv)) + imageCenterY;
        col = (1000.0 * pt.x / (pt.z * focalInv)) + imageCenterX;
}


void
PCLHelperFuns::ID2Img(const pcl::PointIndices::Ptr &ptIdxs, const cv::Mat &ImgOrig, cv::Mat &imOut)
{

    // Find max and min positions of the indices in the original image.
    int colMin = 1000, rowMin = 1000;
    int colMax = 0, rowMax = 0;

    for(uint idx=0; idx!=ptIdxs->indices.size(); idx++)
    {
    std::pair<int,int> pos = PCLHelperFuns::arIdx2MatPos(ptIdxs->indices[idx],640);
    assert(pos.first<480 && pos.second<640);
    int row = pos.first;
    int col = pos.second;

    if (col < colMin)
            colMin = col;
    if (col >= colMax)
            colMax = col;
    if (row < rowMin)
            rowMin = row;
    if (row >= rowMax)
            rowMax = row;
    }


    // Extract rect of interest
    int rx = colMin - 0;
    int ry = rowMin - 0;
    int rWidth = colMax - colMin + 0;
    int rHeight = rowMax - rowMin + 0;
    //  Check that our enlarged ROI is not outside of the image.
    if (rx < 0)
            rx = 0;
    if (ry < 0)
            ry = 0;
    if ((rx + rWidth) > 639)
            rWidth = 639 - rx;
    if ((ry + rHeight) > 479)
            rHeight = 479 - ry;
    cv::Mat r1(ImgOrig, cv::Rect(rx, ry, rWidth, rHeight));
    r1.copyTo(imOut);
}

// Converts between array and matrix position
std::pair<int,int>
PCLHelperFuns::arIdx2MatPos(const int &idx, const int &imWidth)
{
  std::pair<int,int> pos;
  // Row
  pos.first = idx / imWidth;
  // Col
  pos.second = idx % imWidth;

  return pos;
}





/* Iterates through the given point cloud. For each point it moves the vector to
 * the coordinate
 * system of box. It then projects that vector onto the axes of the cube and
 * sees if the point is
 * inside cube.
 * of the
 */
void PCLHelperFuns::selectPointsInCube(const PC::Ptr cloud, const cuboid &cube,
                                       pcl::PointIndices::Ptr indPtr) {
        for (PC::iterator iter = cloud->begin(); iter != cloud->end(); iter++) {
                Eigen::VectorXf p = iter->getVector3fMap(); // p is a column vector
                if (cube.isPtInCuboid(p))
                        indPtr->indices.push_back(iter - cloud->begin());
        }
}

void PCLHelperFuns::colorIndices(PC::Ptr &cloud,
                                 const pcl::PointIndices::Ptr &indPtr,
                                 const V3f &rgb) {
        // Set circle inliers to the color of whatever RGB
        std::vector<int>::iterator iter = indPtr->indices.begin();
        for (; iter != indPtr->indices.end(); ++iter) {
                cloud->points[*iter].r = rgb(0);
                cloud->points[*iter].g = rgb(1);
                cloud->points[*iter].b = rgb(2);
        }
}

void PCLHelperFuns::convEigen2PCL(const V3f &eigVec, PointT &pt) {
        pt.getVector3fMap() = eigVec;
}

void PCLHelperFuns::mergePointIndices(pcl::PointIndices::Ptr &pis1,
                                      const pcl::PointIndices::Ptr &pis2) {
        // Add point indices from vector 2 to vector 1.
        std::vector<int>::iterator valPtr = pis2->indices.begin();
        for (; valPtr != pis2->indices.end(); valPtr++)
                pis1->indices.push_back(*valPtr);

        // Remove duplicate indices
        std::sort(pis1->indices.begin(), pis1->indices.end());
        pis1->indices.erase(unique(pis1->indices.begin(), pis1->indices.end()),
                            pis1->indices.end());
}

/* Computes the points on the object inside the grasp cuboid
 *
 * Input:
 *  cloud - Segmented scene containing only object.
 *  gc - Cuboid defining the grasp
 * Output:
 *  gCloudSceneIdxs - Point indices for the grasped part of the object.
 */
void PCLHelperFuns::computePointsInsideBoundingBox(
        const PC::Ptr &cloud, const cuboid &bb, pcl::PointIndices::Ptr &pIdxs) {
        // Reset idxs
        pIdxs->indices.clear();

        for (int iPt = 0; iPt != cloud->size(); ++iPt) {
                if (bb.isPtInCuboid(cloud->points[iPt].getVector3fMap()))
                        pIdxs->indices.push_back(iPt);
        }
}

/** Checks if there are points in both sides of the cuboid **/
bool PCLHelperFuns::arePointsInsideBoundingBoxEvenlyDistributed(
        const PC::Ptr &cloud, const cuboid &bb) {
        float Nn2left = 0;
        float Nn2right = 0;
        float Nn3left = 0;
        float Nn3right = 0;

        for (int iPt = 0; iPt != cloud->size(); ++iPt) {
                if (bb.isPtInCuboid(cloud->points[iPt].getVector3fMap())) {
                        V3f ptProj =
                                bb.axisRotMat * (cloud->points[iPt].getVector3fMap() - bb.transVec);
                        if (ptProj(1) > 0)
                                Nn2left++;
                        else
                                Nn2right++;

                        if (ptProj(2) > 0)
                                Nn3left++;
                        else
                                Nn3right++;
                }
        }

        // std::cout << Nleft << " " << Nright << std::endl;

        if (Nn2left == 0 || Nn2right == 0 || Nn3left == 0 || Nn3right == 0)
                return false;

        double ratioN2 = Nn2left / Nn2right;
        double ratioN3 = Nn3left / Nn3right;

        if (ratioN2 > 0.8 || ratioN2 < 0.2 || ratioN3 > 0.8 || ratioN3 < 0.2)
                return false;

        return true;
}

void PCLHelperFuns::computePointCloudBoundingBox(const PC::Ptr &cloud,
                                                 cuboid &cube) {
        // compute principal direction
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
                covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
        eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

        // move the points to the that reference frame
        Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
        p2w.block<3, 3>(0, 0) = eigDx.transpose();
        p2w.block<3, 1>(0, 3) = -1.f * (p2w.block<3, 3>(0, 0) * centroid.head<3>());
        PC cPoints;
        pcl::transformPointCloud(*cloud, cPoints, p2w);

        PointT min_pt, max_pt;
        pcl::getMinMax3D(cPoints, min_pt, max_pt);
        V3f mean_diag = 0.5f * (max_pt.getVector3fMap() + min_pt.getVector3fMap());

        // final transform
        Eigen::Quaternionf qfinal(eigDx);
        qfinal.normalize();
        V3f tfinal = eigDx * mean_diag + centroid.head<3>();

        cube.axisMat = eigDx;
        cube.transVec = tfinal;
        cube.quartVec = qfinal;

        cube.width = max_pt.x - min_pt.x;
        cube.height = max_pt.y - min_pt.y;
        cube.depth = max_pt.z - min_pt.z;

        cube.x_min = -0.5f * cube.width;
        cube.x_max = 0.5f * cube.width;
        cube.y_min = -0.5f * cube.height;
        cube.y_max = 0.5f * cube.height;
        cube.z_min = -0.5f * cube.depth;
        cube.z_max = 0.5f * cube.depth;

        cube.setAxisRotMat();

        // Eigen::Matrix3f mat = qfinal.toRotationMatrix();
        // Eigen::Matrix3f axisMatrix = Eigen::Matrix3f::Identity();
        // cube.axisMat.col(0) = qfinal._transformVector(axisMatrix.col(0));
        // cube.axisMat.col(1) = qfinal._transformVector(axisMatrix.col(1));
        // cube.axisMat.col(2) =
        // qfinal._transformVector(axisMatrix.col(2));//axisvector * mat;
}

void PCLHelperFuns::fitCuboid2PointCloudPlane(const PC::Ptr &cloudOrig,
                                              const V3f &tableNrml, cuboid &c) {
        //    PCN::Ptr cloudNormals(new PCN);
        //    PCLHelperFuns::computeCloudNormals(cloud,0.03,cloudNormals);

        PC::Ptr cloud(new PC);
        pcl::copyPointCloud(*cloudOrig, *cloud);

        // Get normal and create other two normals
        V3f n1 = tableNrml;
        n1.normalize();
        Eigen::Vector3f n2(-n1(1), n1(0), 0);
        n2.normalize();
        Eigen::Vector3f n3 = n1.cross(n2);
        n3.normalize();

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);
        V3f plPt = centroid.head<3>();

        /***** CREATE CUBOID ******/

        // Create and assign the axis matrix
        Eigen::Matrix3f cuboidAxis;
        cuboidAxis.col(0) = n1;
        cuboidAxis.col(1) = n2;
        cuboidAxis.col(2) = n3;

        // Remove normal part of points values i.e. project onto plane
        for (PC::iterator cldIter = cloud->begin(); cldIter != cloud->end();
             ++cldIter) {
                V3f pt = cldIter->getVector3fMap();
                cldIter->getVector3fMap() = pt - n1 * n1.dot(pt);
        }

        // Compute principal directions
        pcl::compute3DCentroid(*cloud, centroid);
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
                covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
        eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

        // std::cout << "The eigenvalues of A are:" << std::endl <<
        // eigen_solver.eigenvalues() << std::endl;

        // Compute new normals
        n2 = eigDx.col(2);
        n2.normalize();
        n3 = n1.cross(n2);
        n3.normalize();
        cuboidAxis.col(0) = n1;
        cuboidAxis.col(1) = n2;
        cuboidAxis.col(2) = n3;

        cloud->clear();
        pcl::copyPointCloud(*cloudOrig, *cloud);
        pcl::compute3DCentroid(*cloud, centroid);

        // 3. Move the points to the new reference frame
        Eigen::Matrix4f p3w(Eigen::Matrix4f::Identity());
        p3w.block<3, 3>(0, 0) = cuboidAxis.transpose();
        p3w.block<3, 1>(0, 3) = -1.f * (p3w.block<3, 3>(0, 0) * centroid.head<3>());
        PC::Ptr cPoints(new PC);
        pcl::transformPointCloud(*cloud, *cPoints, p3w);

        // 5. Get min max points
        PointT min_pt, max_pt;
        pcl::getMinMax3D(*cPoints, min_pt, max_pt);
        V3f mean_diag = 0.5f * (max_pt.getVector3fMap() + min_pt.getVector3fMap());
        // mean_diag(0) = 0;

        // final transform
        V3f transVec = cuboidAxis * mean_diag + centroid.head<3>();

        // 6. Add values to cuboid
        c.transVec = transVec;
        Eigen::Quaternionf qfinal(cuboidAxis);
        qfinal.normalize();
        c.quartVec = qfinal;
        c.axisMat = cuboidAxis;

        c.width = max_pt.x - min_pt.x;
        c.height = max_pt.y - min_pt.y;
        c.depth = max_pt.z - min_pt.z;

        c.x_min = -0.5f * c.width;
        c.x_max = 0.5f * c.width;
        c.y_min = -0.5f * c.height;
        c.y_max = 0.5f * c.height;
        c.z_min = -0.5f * c.depth;
        c.z_max = 0.5f * c.depth;
        c.setAxisRotMat();
}

int PCLHelperFuns::computeInliersToCube(const PC::Ptr &cloud, const cuboid &bb,
                                        const double &dThreshold) {

        int counter = 0;

        for (int iPt = 0; iPt != cloud->size(); ++iPt) {
                V3f vec = cloud->points[iPt].getVector3fMap();
                if (bb.isPtInlier(vec, dThreshold))
                        counter++;
        }

        return counter;
}

int PCLHelperFuns::readPCDFile(const std::string &fileName, PC::Ptr &cloud) {

        //  std::cout << "Reading " << fileName << std::endl;
        if (pcl::io::loadPCDFile<PointT>(fileName.c_str(), *cloud) == -1)
        // load the file
        {
                PCL_ERROR("Couldn't read file\n");
                return (-1);
        }
        //  std::cout << "Loaded " << cloud->points.size () << " points." <<
        //  std::endl;

        return 1;
}

void PCLHelperFuns::smoothCloud(PC::Ptr &cloud) {
        PC mls_points;
        pcl::MovingLeastSquares<PointT, PointT> mls;
        mls.setInputCloud(cloud);
        mls.setSearchRadius(0.01);
        mls.setPolynomialFit(true);
        mls.setPolynomialOrder(2);
        // mls.setUpsamplingMethod (pcl::MovingLeastSquares<PointT,
        // PointT>::VOXEL_GRID_DILATION);
        mls.setDilationIterations(0);
        mls.setSqrGaussParam(0.005 * 0.005);
        mls.setDilationVoxelSize(0.001);
        mls.setUpsamplingRadius(0.005);
        mls.setUpsamplingStepSize(0.003);
        mls.process(mls_points);

        // pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        // //pcl::MovingLeastSquares<PointT, pcl::PointNormal> mls;
        // pcl::MovingLeastSquares<PointT, PointT> mls;
        // //pcl::PointCloud<pcl::PointNormal> mls_points;
        // PC mls_points;

        // // Set parameters
        // mls.setInputCloud (cloud);
        // mls.setComputeNormals (true);
        // mls.setPolynomialFit (true);
        // mls.setSearchMethod (tree);
        // mls.setSearchRadius (0.05);
        // // Reconstruct
        // mls.process (mls_points);
        // Copy normal information
        // pcl::copyPointCloud(mls_points, *cldNrmls);
        pcl::copyPointCloud(mls_points, *cloud);
}

void PCLHelperFuns::smoothCloud2(const PC::Ptr &cloudIn, PC::Ptr &cloudOut) {
        pcl::MovingLeastSquares<PointT, PointT> mls;
        mls.setInputCloud(cloudIn);
        mls.setSearchRadius(0.05);
        mls.setPolynomialFit(true);
        mls.setPolynomialOrder(3);
        mls.setUpsamplingMethod(
                pcl::MovingLeastSquares<PointT, PointT>::VOXEL_GRID_DILATION);
        mls.setDilationIterations(1);
        mls.setSqrGaussParam(0.005 * 0.005);
        mls.setDilationVoxelSize(0.001);
        mls.setUpsamplingRadius(0.005);
        mls.setUpsamplingStepSize(0.003);
        mls.process(*cloudOut);

        // Do some cleaning up of noise generated by the upsampling
        V4f min_pt;
        V4f max_pt;
        pcl::getMinMax3D(*cloudIn, min_pt, max_pt);
        pcl::CropBox<PointT> cropBoxFilter;
        cropBoxFilter.setInputCloud(cloudOut);
        cropBoxFilter.setMin(min_pt);
        cropBoxFilter.setMax(max_pt);
        cropBoxFilter.filter(*cloudOut);

        pcl::StatisticalOutlierRemoval<PointT> outlierRemove;
        outlierRemove.setInputCloud(cloudOut);
        outlierRemove.setMeanK(50);
        outlierRemove.setStddevMulThresh(1.0);
        outlierRemove.filter(*cloudOut);
}

void PCLHelperFuns::smoothCloud3(const cv::Mat &im, PC::Ptr &cloud, PCN::Ptr &cloudNormals) {
        // Create a KD-Tree
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

        // Output has the PointNormal type in order to store the normals calculated by
        pcl::PointCloud<pcl::PointXYZRGBNormal> mls_points;

        // Init object (second point type is for the normals, even if unused)
        pcl::MovingLeastSquares<PointT, pcl::PointXYZRGBNormal> mls;
        mls.setComputeNormals(true);
        mls.setUpsamplingMethod(
                pcl::MovingLeastSquares<PointT,
                                        pcl::PointXYZRGBNormal>::VOXEL_GRID_DILATION);
        mls.setDilationVoxelSize(0.0001);
        mls.setPolynomialOrder(4);

        // Set parameters
        mls.setInputCloud(cloud);
        mls.setPolynomialFit(true);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(0.03);
        mls.setPolynomialOrder(3);
        // Reconstruct
        mls.process(mls_points);


        // Copy new cloud and add colors from image
        cloud->clear();
        int row, col;
        PointT pt;
        for (pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator ptIter = mls_points.begin(); ptIter != mls_points.end(); ++ptIter)
        {
                pt.x = ptIter->x;
                pt.y = ptIter->y;
                pt.z = ptIter->z;
                PCLHelperFuns::projectPt2Img(pt, row, col);

                if (row > 0 && row < 480 && col > 0 && col < 640)
                {
                        uint8_t r = (int)im.at<cv::Vec3b>(row, col)[2],
                                g = (int)im.at<cv::Vec3b>(row, col)[1],
                                b = (int)im.at<cv::Vec3b>(row, col)[0]; // Example: Red color
                        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                        pt.rgb = *reinterpret_cast<float *>(&rgb);
                }
                pt.ID = row * 640 + col;
                cloud->push_back(pt);
        }

        // Copy normals
        pcl::copyPointCloud(mls_points, *cloudNormals);

}


void PCLHelperFuns::smoothCloud3(PC::Ptr &cloud, PCN::Ptr &cloudNormals) {
        // Create a KD-Tree
        pcl::search::KdTree<PointT>::Ptr tree(
                new pcl::search::KdTree<PointT>);

        // Output has the PointNormal type in order to store the normals calculated by
        // MLS
        pcl::PointCloud<pcl::PointXYZRGBNormal> mls_points;

        // Init object (second point type is for the normals, even if unused)
        pcl::MovingLeastSquares<PointT, pcl::PointXYZRGBNormal> mls;
        mls.setComputeNormals(true);
        mls.setUpsamplingMethod(
                pcl::MovingLeastSquares<PointT,
                                        pcl::PointXYZRGBNormal>::VOXEL_GRID_DILATION);
        mls.setDilationVoxelSize(0.0001);
        mls.setPolynomialOrder(4);

        // Set parameters
        mls.setInputCloud(cloud);
        mls.setPolynomialFit(true);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(0.03);
        mls.setPolynomialOrder(3);
        // Reconstruct
        mls.process(mls_points);

        // Copy new cloud
        cloud->clear();
        int row, col;
        PointT pt;
        for (pcl::PointCloud<pcl::PointXYZRGBNormal>::iterator ptIter = mls_points.begin(); ptIter != mls_points.end(); ++ptIter)
        {
                pt.x = ptIter->x;
                pt.y = ptIter->y;
                pt.z = ptIter->z;
                pt.ID = row * 640 + col;
                cloud->push_back(pt);
        }
        pcl::copyPointCloud(mls_points, *cloudNormals);
}

void PCLHelperFuns::computeCloudNormals(const PC::Ptr &cloud,
                                        const float &normal_radius,
                                        PCN::Ptr &normals) {
        // PCN::Ptr cloudNormals (new PCN ());
        pcl::NormalEstimationOMP<PointT, PointN> ne;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(normal_radius);
        // ne.setKSearch(50);
        ne.setInputCloud(cloud);
        ne.compute(*normals);
}

/* Compares pixel distances in two images by using the Euclidian distance in the
 * Lab colorspace. Pixels that are further away or closer than a given distance
 * are
 * collected in coordinate vector. Default distance is 20.
 */

void PCLHelperFuns::detectImageDiff(cv::Mat &img1, cv::Mat &img2,
                                    pcl::PointIndices::Ptr &ptsIdx,
                                    std::vector<std::vector<int> > &indices,
                                    const bool diffOrSim, const int &pxDist) {
        if ((img1.rows != img2.rows) || (img1.cols != img2.cols)) {
                std::cout << "Image size differs! Img1: " << img1.size()
                          << " Img2: " << img2.size() << std::endl;
                return;
        }

        // Convert to LAB color space so we can do Euclidian comparisons.
        cv::cvtColor(img1, img1, CV_BGR2Lab);
        cv::cvtColor(img2, img2, CV_BGR2Lab);

        // Do pointer iteration since it is supposed to be F-A-S-T-!
        int Nrows = img1.rows, Ncols = img1.cols;
        cv::Point3_<uchar> *rowPtr1;
        cv::Point3_<uchar> *rowPtr2;
        for (int iRow = 0; iRow != Nrows; iRow++) {
                rowPtr1 = img1.ptr<cv::Point3_<uchar> >(iRow);
                rowPtr2 = img2.ptr<cv::Point3_<uchar> >(iRow);
                for (int iCol = 0; iCol != Ncols; iCol++) {
                        int l1 = rowPtr1[iCol].x, a1 = rowPtr1[iCol].y, b1 = rowPtr1[iCol].z;
                        int l2 = rowPtr2[iCol].x, a2 = rowPtr2[iCol].y, b2 = rowPtr2[iCol].z;

                        double dist =
                                std::pow(l1 - l2, 2) + std::pow(a1 - a2, 2) + std::pow(b1 - b2, 2);
                        dist = std::sqrt(dist);

                        bool saveIdx = false;
                        if (diffOrSim) {
                                if (dist > pxDist)
                                        saveIdx = true;
                        } else {
                                if (dist < pxDist)
                                        saveIdx = true;
                        }

                        if (saveIdx) {
                                std::vector<int> rowColIdx(2, 0);
                                rowColIdx[0] = iRow;
                                rowColIdx[1] = iCol;
                                indices.push_back(rowColIdx);
                                ptsIdx->indices.push_back(iRow * Ncols + iCol);
                        }
                }
        }

        // Convert back from LAB to BGR when exiting.
        cv::cvtColor(img1, img1, CV_Lab2BGR);
        cv::cvtColor(img2, img2, CV_Lab2BGR);
}

/* Computes the difference between two point clouds of the same
 * scene containgin the same amount of indices.
 */
int PCLHelperFuns::detectPointCloudDiff(const PC::Ptr &cloud1,
                                        const PC::Ptr &cloud2,
                                        pcl::PointIndices::Ptr &indices) {
        // Check that clouds are of the same size
        if (cloud1->size() != cloud2->size()) {
                std::cout << "---SIZE DIFFERS!!---" << std::endl;
                return 0;
        }

        for (int pIdx = 0; pIdx != cloud1->size(); pIdx++) {
                V3f vecDiff = cloud1->points[pIdx].getVector3fMap() -
                              cloud2->points[pIdx].getVector3fMap();
                V3f colorDiff(cloud1->points[pIdx].r - cloud2->points[pIdx].r,
                              cloud1->points[pIdx].g - cloud2->points[pIdx].g,
                              cloud1->points[pIdx].b - cloud2->points[pIdx].b);
                cv::Mat c1(1, 1, CV_32FC3, cv::Scalar(cloud1->points[pIdx].r / 255.0,
                                                      cloud1->points[pIdx].g / 255.0,
                                                      cloud1->points[pIdx].b / 255.0));
                cv::Mat c2(1, 1, CV_32FC3, cv::Scalar(cloud2->points[pIdx].r / 255.0,
                                                      cloud2->points[pIdx].g / 255.0,
                                                      cloud2->points[pIdx].b / 255.0));
                cv::cvtColor(c1, c1, CV_RGB2Lab);
                cv::cvtColor(c2, c2, CV_RGB2Lab);
                cv::Vec3f c1Val = c1.at<cv::Vec3f>(0, 0);
                cv::Vec3f c2Val = c2.at<cv::Vec3f>(0, 0);
                float colorDist = cv::norm(c1 - c2);

                double d = vecDiff.norm();
                // double c = colorDiff.norm();
                // printf("CDiff %f\n",colorDist);
                if (d > 0.01 && d < 0.08 && colorDist > 60.0) {
                        indices->indices.push_back(pIdx);
                }
        }

        return 1;
}

/* Computes the difference between two point clouds of the same
 * scene containgin the same amount of indices.
 */
void PCLHelperFuns::detectPointCloudDiffByOrigID(
        const PC::Ptr &cloud1, const PC::Ptr &cloud2,
        pcl::PointIndices::Ptr &indices) {

        int pc1Idx(0), pc2Idx(0);
        PC::const_iterator pc1Iter;
        PC::const_iterator pc2Iter;
        int ptIdx = 0;
        bool ptFound = false;
        for (pc1Iter = cloud1->begin(); pc1Iter != cloud1->end(); ++pc1Iter) {
                ptFound = false;
                for (pc2Iter = cloud2->begin(); pc2Iter != cloud2->end(); ++pc2Iter) {
                        if (pc1Iter->ID == pc2Iter->ID) {
                                ptFound = true;
                                break;
                        }
                }

                if (ptFound) {
                        // printf("Point found.\n");
                        V3f vecDiff = pc1Iter->getVector3fMap() - pc2Iter->getVector3fMap();
                        V3f colorDiff(pc1Iter->r - pc2Iter->r, pc1Iter->g - pc2Iter->g,
                                      pc1Iter->b - pc2Iter->b);
                        cv::Mat c1(1, 1, CV_8UC3, cv::Scalar(pc1Iter->r, pc1Iter->g, pc1Iter->b));
                        cv::Mat c2(1, 1, CV_8UC3, cv::Scalar(pc2Iter->r, pc2Iter->g, pc2Iter->b));
                        cv::cvtColor(c1, c1, CV_RGB2Lab);
                        cv::cvtColor(c2, c2, CV_RGB2Lab);
                        cv::Vec3b c1Val = c1.at<cv::Vec3b>(0, 0);
                        cv::Vec3b c2Val = c2.at<cv::Vec3b>(0, 0);

                        // std::cout<< "cv1mat " << c1 << std::endl;
                        // std::cout<< "cv2mat "<< c2 << std::endl;
                        // std::cout<< "cv1vec " << c1Val << std::endl;
                        // std::cout<< "cv2vec "<< c2Val << std::endl;

                        float colorDist = cv::norm(c1 - c2);
                        // std::cout << "Distance " << colorDist << std::endl;

                        // cv::Mat cHSV1(1,1, CV_32FC3,
                        // cv::Scalar(pc1Iter->r/255.0,pc1Iter->g/255.0,pc1Iter->b/255.0));
                        // cv::Mat cHSV2(1,1, CV_32FC3,
                        // cv::Scalar(pc2Iter->r/255.0,pc2Iter->g/255.0,pc2Iter->b/255.0));

                        cv::Mat cHSV1(1, 1, CV_8UC3,
                                      cv::Scalar(pc1Iter->r, pc1Iter->g, pc1Iter->b));
                        cv::Mat cHSV2(1, 1, CV_8UC3,
                                      cv::Scalar(pc2Iter->r, pc2Iter->g, pc2Iter->b));

                        cv::cvtColor(cHSV1, cHSV1, CV_RGB2YCrCb);
                        cv::cvtColor(cHSV2, cHSV2, CV_RGB2YCrCb);

                        cv::Vec3f c1HSVVal = cHSV1.at<cv::Vec3b>(0, 0);
                        cv::Vec3f c2HSVVal = cHSV2.at<cv::Vec3b>(0, 0);

                        // std::cout<< "c1HSVMat " << cHSV1 << std::endl;
                        // std::cout<< "c2HSVMat " << cHSV2 << std::endl;

                        // std::cout<< "c1HSVVal " << c1HSVVal << std::endl;
                        // std::cout<< "c2HSVVal " << c2HSVVal << std::endl;

                        // if(c2HSVVal(0)<0 || c2HSVVal(0)>100)
                        //  continue;
                        // if(c2HSVVal(1)<133 || c2HSVVal(1)>173)
                        //  continue;
                        // if(c2HSVVal(2)<77 || c2HSVVal(2)>127)
                        // continue;

                        double d = vecDiff.norm();
                        // double c = colorDiff.norm();
                        // printf("CDiff %f\n",colorDist);
                        if (d > 0.01 && d < 0.08) {
                                indices->indices.push_back(ptIdx);
                        }
                        // printf("005\n");
                }
                ptIdx++;
        }

        // std::cout << "DATA03" << std::endl;
}

void PCLHelperFuns::matchIndices2Scene(const PC::Ptr &cluster,
                                       const PC::Ptr &cloud,
                                       pcl::PointIndices::Ptr &ptIdxs) {
        /* This algorithm is an ugly hack. First we pick the first point in
         * the cluster point cloud and start comparing its distance to points in the
         * cloud
         * for the scene. When the distance is zero we know have found the
         * point in the scence and we can then increment the indices in the
         * index vector accordingly.
         */

        /* Variables are:
         * cluster, that is a segmentation of the original point cloud.
         * cloud, the original scene in which the cluster resided.
         * ptIdxs, indices over the cluster, doesn't have to be one to one!
         */
        int pIdx = 0;
        for (int ipnt = 0; ipnt != cluster->points.size(); ipnt++) {
                V3f pt = cluster->points[ipnt].getVector3fMap();
                for (pIdx = 0; pIdx != cloud->size(); pIdx++) {
                        V3f vecDiff = cloud->points[pIdx].getVector3fMap() - pt;
                        if (vecDiff.norm() < 0.001) {
                                ptIdxs->indices.push_back(pIdx);
                                break; // Got the index position so break
                        }
                }
        }
}

void PCLHelperFuns::matchIndices2SceneByOrigID(const PC::Ptr &cluster,
                                               const PC::Ptr &cloud,
                                               pcl::PointIndices::Ptr &ptIdxs) {
        /* This algorithm is an ugly hack. First we pick the first point in
         * the cluster point cloud and start comparing its distance to points in the
         * cloud
         * for the scene. When the distance is zero we know have found the
         * point in the scence and we can then increment the indices in the
         * index vector accordingly.
         */

        /* Variables are:
         * cluster, that is a segmentation of the original point cloud.
         * cloud, the original scene in which the cluster resided.
         * ptIdxs, indices over the cluster, doesn't have to be one to one!
         */
        int pIdx = 0;
        for (int ipnt = 0; ipnt != cluster->points.size(); ipnt++) {
                V3f pt = cluster->points[ipnt].getVector3fMap();
                bool retain = false;
                for (pIdx = 0; pIdx != cloud->size(); pIdx++) {
                        if (cluster->points[ipnt].ID == cloud->points[pIdx].ID) {
                                V3f vecDiff = cloud->points[pIdx].getVector3fMap() - pt;
                                if (vecDiff.norm() < 0.001) {
                                        ptIdxs->indices.push_back(pIdx);
                                        break; // Got the index position so break
                                }
                        }
                }
        }
}

// Finds the index difference between cluster and cloud by finding the cluster's
// first point in the cloud
int PCLHelperFuns::findCloudOffset(const PC::Ptr &cluster,
                                   const PC::Ptr &cloud) {

        /* Variables are:
         * cluster, that is a segmentation of the original point cloud.
         * cloud, the original scene in which the cluster resided.
         * ptIdxs, indices over the cluster, doesn't have to be one to one!
         */
        int pIdx = 0;

        V3f pt = cluster->points[0].getVector3fMap();
        for (; pIdx != cloud->size(); pIdx++) {
                V3f vecDiff = cloud->points[pIdx].getVector3fMap() - pt;
                if (vecDiff.norm() < 0.001)
                        break;  // Got the index position so break
        }
        return pIdx;
}

/* Filters out points in the point cloud that matches the original ID with the
 * indices in pointindices. */
void PCLHelperFuns::filterOutByOrigID(const pcl::PointIndices::Ptr &ptOrigIdxs,
                                      const PC::Ptr &cloudIn,
                                      PC::Ptr &cloudOut) {
        // Extract the original indices
        pcl::PointIndices::Ptr ptIdxs(new pcl::PointIndices);
        size_t j = 0;
        size_t i_last = 0;

        for (std::vector<int>::const_iterator ptIter = ptOrigIdxs->indices.begin();
             ptIter != ptOrigIdxs->indices.end(); ++ptIter) {
                j = 0;
                // Find point with ID in point cloud and add it to the indicies
                for (PC::iterator cloudIter = cloudIn->begin(); cloudIter != cloudIn->end();
                     ++cloudIter) {
                        if (cloudIter->ID == *ptIter) {
                                ptIdxs->indices.push_back(j);
                        }
                        j++;
                }
        }

        // Use PCL to remove them.
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(true);
        extract.setInputCloud(cloudIn);
        extract.setIndices(ptIdxs);
        extract.filter(*cloudOut);
}


void
PCLHelperFuns::filterCloudFromCloud(const PC::Ptr &cloud2Filter,const PC::Ptr &cloud2FilterOut, PC::Ptr &cloudOut, float voxSize)
{

// Create voxel grid
float voxelSize = voxSize; // voxel resolution
pcl::octree::OctreePointCloud<PointT> octree (voxelSize);
// Set input point cloud (via Boost shared pointers):
octree.setInputCloud (cloud2FilterOut);
octree.addPointsFromInputCloud ();
// Define octree bounding box (optional):
// calculate bounding box of input cloud
// octree.defineBoundingBox ();
// manually define bounding box
// octree.defineBoundingBox (minX, minY, minZ, maxX, maxY, maxZ);
// Add points from input cloud to octree:
// octree.addPointsFromInputCloud ();
// Delete octree data structure:
// (pushes allocated nodes to memory pool!)
// octree.deleteTree ();

// For each point in cloud check if point exist




double X,Y,Z;
bool occuppied;
pcl::PointIndices::Ptr ptIdxs(new pcl::PointIndices);
uint j = 0;
for (PC::iterator ptIter = cloud2Filter->begin(); ptIter != cloud2Filter->end(); ++ptIter)
        {

                X = ptIter->x;
                Y = ptIter->y;
                Z = ptIter->z;
                if(octree.isVoxelOccupiedAtPoint(*ptIter))
                {
                    ptIdxs->indices.push_back(j);
                }
                j++;
        }



// occuppied = octree.isVoxelOccupiedAtPoint (X, Y, Z);
// Get center points of all occupied voxels:
// (voxel grid filter/downsampling)
// std::vector<PointXYZ> pointGrid;
// octree.getOccupiedVoxelCenters (pointGrid);
// Query points within a voxel:
// std::vector<int> pointIdxVec;
// octree.voxelSearch (searchPoint, pointIdxVec);
// Delete voxel:
// pcl::PointXYZ point_arg( 1.0, 2.0, 3.0 );
// octree.deleteVoxelAtPoint ( point );

        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(true);
        extract.setInputCloud(cloud2Filter);
        extract.setIndices(ptIdxs);
        extract.filter(*cloudOut);
}




/* Extracts points in the point cloud that matches the original ID with the
 * indices in pointindices. */
void PCLHelperFuns::filterByOrigID(const PC::Ptr &cloudIn,
                                   const pcl::PointIndices::Ptr &ptIdxs,
                                   PC::Ptr &cloudOut) {
        size_t j = 0;
        size_t i_last = 0;

        for (std::vector<int>::const_iterator ptIter = ptIdxs->indices.begin();
             ptIter != ptIdxs->indices.end(); ++ptIter) {
                bool pushed = false;
                j = 0;
                // Find point with ID in point cloud and copy it
                for (PC::iterator cloudIter = cloudIn->begin(); cloudIter != cloudIn->end();
                     ++cloudIter) {
                        if (cloudIter->ID == *ptIter) {
                                cloudOut->push_back(*cloudIter);
                                pushed = true;
                                i_last = j;
                                break;
                        }
                        j++;
                }
        }

        cloudOut->height = 1;
        cloudOut->width = static_cast<unsigned int>(ptIdxs->indices.size());

        // Removing bad points => dense (note: 'dense' doesn't mean 'organized')
        cloudOut->is_dense = true;
}

void PCLHelperFuns::samplePlane(const PC::Ptr &cloud, V4f &planeParams,
                                V3f &planePt) {

        // Create uniform distribution over point cloud indices
        boost::random::random_device rd;
        boost::random::mt19937 gen(rd());
        int Npts = cloud->size();
        boost::random::uniform_int_distribution<> dis(0, Npts);

        // Sample 3 random points in cloud
        int pt1Idx = dis(gen);
        int pt2Idx = dis(gen);
        int pt3Idx = dis(gen);

        // Turn into Eigen Vectors
        V3f pt1 = cloud->points[pt1Idx].getArray3fMap();
        V3f pt2 = cloud->points[pt2Idx].getArray3fMap();
        V3f pt3 = cloud->points[pt3Idx].getArray3fMap();

        // Compute plane params
        V3f vec1 = pt2 - pt1;
        V3f vec2 = pt3 - pt1;
        V3f plnNormal = vec1.cross(vec2);
        plnNormal.normalize();

        planeParams.head<3>() = plnNormal;
        planeParams(3) = -1.f * (planeParams.head<3>()).dot(pt1);
        planePt = (pt1 + pt2 + pt3) / 3;
}

void PCLHelperFuns::filterOutSmallClusters(PC::Ptr &cloud,
                                           const float &clusterTolerance,
                                           const int &minClusterSize) {

        // Cluster the points on the plane. This command finds all clusters with more
        // points than setMaxClusterSize
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(clusterTolerance); // 0.01 = 1cm
        ec.setMinClusterSize(minClusterSize);
        ec.setMaxClusterSize(7500);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Find the biggest cluster index
        int maxClusterIdx = 0;
        for (int i_cluster = 0; i_cluster < cluster_indices.size(); i_cluster++) {
                //    std::cout << "Cluster size: " <<
                //    cluster_indices[i_cluster].indices.size() << std::endl;
                if (cluster_indices[i_cluster].indices.size() >
                    cluster_indices[maxClusterIdx].indices.size()) {
                        maxClusterIdx = i_cluster;
                }
        }

        // MyHelperFuns::printString("Number of large clusters found:
        // "+std::to_string(cluster_indices.size()),printMsg);

        if (cluster_indices.size() == 0) {
                return;
        }

        pcl::PointIndices::Ptr clusterPtr(
                new pcl::PointIndices(cluster_indices[maxClusterIdx]));

        /********************** Filter out everything else **********************/

        // Filter out everything but the points in the biggest cluster.
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(false);
        extract.setInputCloud(cloud);
        extract.setIndices(clusterPtr);
        extract.filter(*cloud);
}

/** This only works for Kinect clouds **/
void PCLHelperFuns::cloud2img(const PC::Ptr &cloud, const bool &hasColor,
                              cv::Mat &img) {
        cv::Mat pcImg = cv::Mat::zeros(480, 640, CV_8UC3);
        double focalInv = 1000.0 / 525.0;

        int xMin = 640;
        int xMax = 0;
        int yMin = 480;
        int yMax = 0;

        if (hasColor) {

                // Make whole image color of blue so that we can see lost pixels
                pcImg.setTo(cv::Scalar(255, 0, 0));

                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
                for (PC::const_iterator iter = cloud->begin(); iter != cloud->end();
                     ++iter) {
                        double xm = iter->x;
                        double ym = iter->y;
                        double zm = iter->z;
                        int xPos = (1000.0 * xm / (zm * focalInv)) + 320;
                        int yPos = (1000.0 * ym / (zm * focalInv)) + 240;

                        pcImg.at<cv::Vec3b>(yPos, xPos)[0] = iter->b; // BGR 2 RGB
                        pcImg.at<cv::Vec3b>(yPos, xPos)[1] = iter->g;
                        pcImg.at<cv::Vec3b>(yPos, xPos)[2] = iter->r;

                        if (xPos < xMin)
                                xMin = xPos;
                        if (xPos > xMax)
                                xMax = xPos;
                        if (yPos < yMin)
                                yMin = yPos;
                        if (yPos > yMax)
                                yMax = yPos;
                }
        } else {
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
                for (PC::const_iterator iter = cloud->begin(); iter != cloud->end();
                     ++iter) {
                        double xm = iter->x;
                        double ym = iter->y;
                        double zm = iter->z;
                        int xPos = (1000.0 * xm / (zm * focalInv)) + 320;
                        int yPos = (1000.0 * ym / (zm * focalInv)) + 240;

                        pcImg.at<cv::Vec3b>(yPos, xPos)[0] = 255; // BGR 2 RGB
                        pcImg.at<cv::Vec3b>(yPos, xPos)[1] = 255;
                        pcImg.at<cv::Vec3b>(yPos, xPos)[2] = 255;

                        if (xPos < xMin)
                                xMin = xPos;
                        if (xPos > xMax)
                                xMax = xPos;
                        if (yPos < yMin)
                                yMin = yPos;
                        if (yPos > yMax)
                                yMax = yPos;
                }
        }

        // Crop img
        pcImg(cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin)).copyTo(img);
}
