#include "scenesegmentation.h"

// std
#include <cmath>
#include <ctime>
#include <iostream>
#include <stdexcept>
// #include <fstream>

// Open CV
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

// PCL headers
//#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/impl/sac_model_circle3d.hpp>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/mls.h>

// Class headers
#include <boosthelperfuns.h>
#include <myhelperfuns.h>
#include <opencvhelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>

// Constructor...
SceneSegmentation::SceneSegmentation()
        : fileName(""), printMsg(true), isScene3DSegemented(false),
        isScene2DSegemented(false), cloudScene(new PC), cloudSegmented(new PC),
        subSampledSegmentedCloud(new PC), cloudSegmentedNormals(new PCN),
        subSampledSegmentedCloudNormals(new PCN) {
}

SceneSegmentation::~SceneSegmentation()
{
        deleteTmpImFiles();
}

int SceneSegmentation::setInputSource(const std::string &fileName) {
        if (!BoostHelperFuns::fileExist(fileName))
                throw std::runtime_error("Found no pcd file!");

        // Read point cloud to sceneCloud variable
        readPointCloudFromFile(fileName, cloudScene);

        std::cout << "Read point cloud with " << cloudScene->points.size() << " points"<< '\n';

        // Init the segmented image to zeros
        imgSegmented = cv::Mat::zeros(480, 640, CV_8UC3);

        // Get filename without extension
        int lastindex = fileName.find_last_of(".");
        rawfileName = fileName.substr(0, lastindex);

        // Load image of the scene(BGR!!)
        if (!BoostHelperFuns::fileExist(rawfileName + ".png"))
                throw std::runtime_error("Found no png file!");
        img = cv::imread((rawfileName + ".png"));

        OpenCVHelperFuns::printTypeInfo("Img", img);

        return 1;
}

int SceneSegmentation::readPointCloudFromFile(const std::string &fName,
                                              PC::Ptr &inputCloud) {
        PCLHelperFuns::readPCDFile(fName, inputCloud);
        inputCloud->height = 480;
        inputCloud->width = 640;

        // Set point id, that is, the original point cloud index
        for (size_t iPt = 0; iPt < inputCloud->points.size(); ++iPt)
                inputCloud->points[iPt].ID = iPt;

        // Remove NaNs
        inputCloud->is_dense = false;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*inputCloud, *inputCloud, indices);
        inputCloud->is_dense = true;
        return 1;
}

/* Takes the given point cloud and removes particles further away than a
 * distance r.
 * From the remaining points it segments out a plane. It then removes the points
 * of the plane. The remaining points are the particles of objects standing on
 * the plane.
 */
void SceneSegmentation::segmentPointCloud() {

        // TMP storage clouds
        PC::Ptr cloud(new PC);

        // Make a deep copy of original point cloud
        pcl::copyPointCloud(*cloudScene, *cloud);

        MyHelperFuns::printString("Starting point cloud segementation", printMsg);
        MyHelperFuns::printString("PointCloud has: " +
                                  MyHelperFuns::toString(cloud->points.size()) +
                                  " data points.",
                                  printMsg);

        /********************** Filter out distant points **********************/
        // z - camera direction
        // x - horizontal
        // y - vertical

        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0., 1.3);
        pass.filter(*cloud);

        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-8., 1.0);
        pass.filter(*cloud);

        pass.setInputCloud(cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-8., 1.0);
        pass.filter(*cloud);

        MyHelperFuns::printString("PointCloud after removing distant points has: " +
                                  MyHelperFuns::toString(cloud->size()) +
                                  " data points.",
                                  printMsg);

        pcl::VoxelGrid<PointT> vgss;
        vgss.setInputCloud(cloud);
        vgss.setLeafSize(0.003f, 0.003f, 0.003f);
        vgss.filter(*subSampledSegmentedCloud);

        //    pcl::StatisticalOutlierRemoval<PointT> sor;
        //    sor.setInputCloud (subSampledSegmentedCloud);
        //    sor.setMeanK (50);
        //    sor.setStddevMulThresh (1.0);
        //    sor.filter(*subSampledSegmentedCloud);
        //    subSampledSegmentedCloud->is_dense = false;
        //    std::vector<int> indices;
        //    pcl::removeNaNFromPointCloud(*subSampledSegmentedCloud,*subSampledSegmentedCloud,
        //    indices);
        //    subSampledSegmentedCloud->is_dense = true;

        PCLHelperFuns::computeCloudNormals(subSampledSegmentedCloud, 0.03,
                                           subSampledSegmentedCloudNormals);
        //    PCLHelperFuns::computeCloudNormals(cloud,0.03,cloudSegmentedNormals);

        /********************** Find the plane in the scene **********************/
        //    V3f plNrml(0.114223,-0.95439,-0.275849);
        // First simple plane localization
        pcl::SACSegmentationFromNormals<PointT, PointN> seg;
        pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr pmPlane(new pcl::ModelCoefficients);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(200);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(subSampledSegmentedCloud);
        seg.setInputNormals(subSampledSegmentedCloudNormals);
        // Obtain the plane inliers and plane parameters
        seg.segment(*inliers_plane, *pmPlane);

        if (pmPlane->values[2] > 0.0)
                plCf = -V4f(pmPlane->values[0], pmPlane->values[1], pmPlane->values[2],
                            pmPlane->values[3]);
        else
                plCf = V4f(pmPlane->values[0], pmPlane->values[1], pmPlane->values[2],
                           pmPlane->values[3]);

        // Find the actual plane inliers by fitting a plane with normal of the plane
        // we found
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
        seg.setNormalDistanceWeight(0.5);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(100);
        seg.setDistanceThreshold(0.01);
        seg.setAxis(plCf.head<3>().normalized());
        seg.setEpsAngle(DEG2RAD(5));
        seg.setInputCloud(subSampledSegmentedCloud);
        seg.setInputNormals(subSampledSegmentedCloudNormals);
        // Obtain the plane inliers and plane parameters
        seg.segment(*inliers_plane, *pmPlane);

        // Find the convex hull of the table and points above it by adding a
        // a point above the plane and then removing all points outside the hull.

        // Find x,y,z filters
        PC::Ptr tableCld(new PC);
        pcl::ExtractIndices<PointT> extract_table;
        extract_table.setNegative(false);
        extract_table.setInputCloud(subSampledSegmentedCloud);
        extract_table.setIndices(inliers_plane);
        extract_table.filter(*tableCld);

        // Find center and add a new point 50cm up from the table normal
        V4f centroid;
        pcl::compute3DCentroid(*tableCld, centroid);
        V3f tableUpMax = centroid.head<3>() + 1.5 * plCf.head<3>().normalized();

        PointT P;
        P.x = tableUpMax[0];
        P.y = tableUpMax[1];
        P.z = tableUpMax[2];
        P.r = 255;
        P.g = 0;
        P.b = 0;
        tableCld->push_back(P);

        V4f minPt;
        V4f maxPt;
        pcl::getMinMax3D(*tableCld, minPt, maxPt);

        std::vector<int> indices;
        pcl::getPointsInBox(*cloud, minPt, maxPt, indices);

        // Do the same for the scence
        pass.setInputCloud(cloudScene);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(minPt[0], maxPt[0]);
        pass.filter(*cloudScene);

        pass.setInputCloud(cloudScene);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(minPt[1], maxPt[1]);
        pass.filter(*cloudScene);

        pass.setInputCloud(cloudScene);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt[2], maxPt[2]);
        pass.filter(*cloudScene);

        /************ Remove plane inliers and below plane points *************/

        pcl::PointIndices::Ptr belowPlane(new pcl::PointIndices);

        //    #pragma omp parallel for num_threads(4)
        for (uint idx = 0; idx != cloud->size(); idx++) {
                V4f p = cloud->points[idx].getVector4fMap();
                p[3] = 1;
                double dp = plCf.dot(p);

                if (dp < 0.01 || dp > 0.50)
                        belowPlane->indices.push_back(idx);
        }

        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(true);
        extract.setInputCloud(cloud);
        extract.setIndices(belowPlane);
        extract.filter(*cloudSegmented);

        // Compute a convex hull and filter things out
        pcl::ConvexHull<PointT> hull;
        hull.setInputCloud(tableCld);
        hull.setDimension(3);
        std::vector<pcl::Vertices> polygons;

        PC::Ptr surface_hull(new PC);
        hull.reconstruct(*surface_hull, polygons);
        pcl::CropHull<PointT> bb_filter;

        bb_filter.setDim(3);
        bb_filter.setInputCloud(cloudSegmented);
        bb_filter.setHullIndices(polygons);
        bb_filter.setHullCloud(surface_hull);
        bb_filter.filter(*cloudSegmented);

        //    clusterCloud(cloudSegmented,centroid);

        MyHelperFuns::printString(
                "PointCloud after removing below plane points has: " +
                MyHelperFuns::toString(cloudSegmented->size()) + " data points.",
                printMsg);

        // Remove outliers of the plane some remaining points of the plane.
        pcl::StatisticalOutlierRemoval<PointT> outlierRemove;
        outlierRemove.setInputCloud(cloudSegmented);
        outlierRemove.setMeanK(50);
        outlierRemove.setStddevMulThresh(1.0);
        outlierRemove.filter(*cloudSegmented);

        // PCLHelperFuns::smoothCloud(cloudSegmented);
        // PCLHelperFuns::computeCloudNormals(cloudSegmented, 0.03,
        //                                    cloudSegmentedNormals);

        PCLHelperFuns::smoothCloud3(img,cloudSegmented,cloudSegmentedNormals);

        setIDs(cloudSegmented);

        vgss.setInputCloud(cloudSegmented);
        vgss.setLeafSize(0.003f, 0.003f, 0.003f);
        vgss.filter(*subSampledSegmentedCloud);
        PCLHelperFuns::computeCloudNormals(subSampledSegmentedCloud, 0.01,
                                           subSampledSegmentedCloudNormals);

        isScene3DSegemented = true;
}

void SceneSegmentation::setupCloud() {
        cv::imwrite(rawfileName + ".ppm", img);
        cv::imshow( "GaussianBlur", img );
        cv::waitKey();
        pcl::copyPointCloud(*cloudScene, *cloudSegmented);
        PCLHelperFuns::smoothCloud3(img,cloudSegmented,cloudSegmentedNormals);
        // PCLHelperFuns::computeCloudNormals(cloudSegmented, 0.03,
        //                                    cloudSegmentedNormals);
        Eigen::VectorXf offset = EigenHelperFuns::readVec(rawfileName + "loc.txt");
        std::cout << "Offset " << offset[0] << " " << offset[1] << '\n';

        int imcol = img.cols;
        // Copy image
        cv::Mat img_new = cv::Mat::zeros(480, 640, CV_8UC3);
        // Copy cropped image into position
        img.copyTo(img_new(cv::Rect(offset[0],offset[1],img.cols,img.rows)));
        img_new.copyTo(imgSegmented);
        img_new.copyTo(img);

        // cv::imwrite(rawfileName + ".ppm", img);
        cv::imshow( "GaussianBlur", imgSegmented );
        cv::imshow( "GaussianBlur3", img );
        cv::waitKey();

        //    PointT p = cloudSegmented->points[iter];
        //    // Compute approximate coordinates of pixel position of segmented
        //    object
        //    int col = (1000.0 * p.x / (p.z * focalInv)) + imageCenterX;
        //    int row = (1000.0 * p.y / (p.z * focalInv)) + imageCenterY;

        double focal = 570.3;
        int imageCenterX = 640/2;
        int imageCenterY = 480/2;
        cv::Mat imgCloud = cv::Mat::zeros(480, 640, CV_8UC1);
        for (size_t iPt = 0; iPt < cloudSegmented->points.size(); ++iPt)
        {
                int col = focal * (cloudSegmented->points[iPt].x / cloudSegmented->points[iPt].z) + imageCenterX;
                int row = focal * (cloudSegmented->points[iPt].y / cloudSegmented->points[iPt].z) + imageCenterY;

                //  col = (pt.x/pt.z)*focal - locvals[0] - depthImMat.cols + cCol
                //  row -= offset[1];
                //  col -= offset[0];imgGrad2_.at<uchar>(pos.first,pos.second);
                imgCloud.at<uchar>(row,col) = 125;
                cloudSegmented->points[iPt].ID = row*640 + col;
                // std::cout << "Point ID " << iPt << " " << cloudSegmented->points[iPt].ID << '\n';
                std::cout << "Pos object(id,row,col) " << cloudSegmented->points[iPt].ID << " " << row << " " << col << '\n';
        }
        offset_ = offset[0] * 640 + offset[1];
        //
        //  int imageCenterX = 640/2;
        //  int imageCenterY = 480/2;

        cv::imshow( "imgCloud", imgCloud );
        cv::waitKey();
        // std::cout << "Ponint id at 0" << cloudSegmented->points[0].ID << '\n';
        pcl::VoxelGrid<PointT> vgss;
        vgss.setInputCloud(cloudSegmented);
        vgss.setLeafSize(0.003f, 0.003f, 0.003f);
        vgss.filter(*subSampledSegmentedCloud);
        PCLHelperFuns::computeCloudNormals(subSampledSegmentedCloud, 0.01,
                                           subSampledSegmentedCloudNormals);

        // cv::imwrite((rawfileName + "_seg.png"), img);
        // cv::imwrite((rawfileName + ".ppm"), img);
        isScene2DSegemented = true;
}

void SceneSegmentation::setIDs(PC::Ptr &cloud)
{
        double focal = 525.;
        int imageCenterX = 640/2;
        int imageCenterY = 480/2;
        cv::Mat imgCloud = cv::Mat::zeros(480, 640, CV_8UC1);
        for (size_t iPt = 0; iPt < cloud->points.size(); ++iPt)
        {
                int col = focal * (cloud->points[iPt].x / cloud->points[iPt].z) + imageCenterX;
                int row = focal * (cloud->points[iPt].y / cloud->points[iPt].z) + imageCenterY;
                cloud->points[iPt].ID = row*640 + col;
                // std::cout << "Point ID " << iPt << " " << cloudSegmented->points[iPt].ID << '\n';
                // std::cout << "Pos object(id,row,col) " << cloudSegmented->points[iPt].ID << " " << row << " " << col << '\n';
        }
}

int SceneSegmentation::rmHalfPC(const int &partId) {
        // PartID is the part to keep
        // 0 - left half
        // 1 - right half
        // 2 - upper half
        // 3 - lower half

        // Locate image row and col max in the cloud
        int colMin = 1000, rowMin = 1000;
        int colMax = 0, rowMax = 0;
        std::vector<int> pxPos(2);
        for (uint idx = 0; idx != cloudSegmented->size(); idx++) {
                PCLHelperFuns::pt2px(640, cloudSegmented->points[idx].ID, pxPos);
                if (pxPos[0] < rowMin)
                        rowMin = pxPos[0];
                if (pxPos[0] > rowMax)
                        rowMax = pxPos[0];
                if (pxPos[1] < colMin)
                        colMin = pxPos[1];
                if (pxPos[1] > colMax)
                        colMax = pxPos[1];
        }

        // Compute vertical and horizontal split
        int colSpltPt = (colMax - colMin) / 2 + colMin;
        int rowSpltPt = (rowMax - rowMin) / 2 + rowMin;

        // Filter out all points but wanted split
        //    std::cout << colSpltPt << ", " << rowSpltPt << std::endl;
        pcl::PointIndices::Ptr pts2Save(new pcl::PointIndices);
        for (uint idx = 0; idx != cloudSegmented->size(); idx++) {
                PCLHelperFuns::pt2px(640, cloudSegmented->points[idx].ID, pxPos);

                int row = pxPos[0];
                int col = pxPos[1];
                //        std::cout << row << ", " << col << std::endl;
                if (partId == 0) // Right part
                {
                        if (col > colSpltPt) {
                                pts2Save->indices.push_back(idx);
                        }
                } else if (partId == 1) // Left part
                {
                        if (col <= colSpltPt) {
                                pts2Save->indices.push_back(idx);
                        }
                } else if (partId == 2) // Lower part
                {
                        if (row <= rowSpltPt) {
                                pts2Save->indices.push_back(idx);
                        }
                } else if (partId == 3) // Upper part
                {
                        if (row > rowSpltPt) {
                                pts2Save->indices.push_back(idx);
                        }
                }
        }

        // Remove all points that we don't want
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(true);
        extract.setInputCloud(cloudSegmented);
        extract.setIndices(pts2Save);
        extract.filter(*cloudSegmented);

        PCLHelperFuns::computeCloudNormals(cloudSegmented, 0.03,
                                           cloudSegmentedNormals);
        pcl::VoxelGrid<PointT> vgss;
        vgss.setInputCloud(cloudSegmented);
        vgss.setLeafSize(0.003f, 0.003f, 0.003f);
        vgss.filter(*subSampledSegmentedCloud);
        PCLHelperFuns::computeCloudNormals(subSampledSegmentedCloud, 0.01,
                                           subSampledSegmentedCloudNormals);

        return 1;
}

void SceneSegmentation::clusterCloud(PC::Ptr &cloud, const V4f &tblcentroid) {

        /* Since our asumptions is that there is only one object on the surface we
         * can do euclidian clustering and pick out the biggest cluster and throw
         * away everything else as noise or outliers.
         */

        //     Cluster the points on the plane. This command finds all clusters with
        //     more
        //     points than setMaxClusterSize
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(0.10); // 0.01 = 1cm
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(75000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        std::cout << "Cluster size: " << cluster_indices.size() << std::endl;
        // Find the biggest cluster index
        int maxClusterIdx = 0;
        double minDist = 10000000000.;
        for (uint i_cluster = 0; i_cluster < cluster_indices.size(); i_cluster++) {
                pcl::PointIndices::Ptr clusterPtr(
                        new pcl::PointIndices(cluster_indices[i_cluster]));
                PC::Ptr tmpcld(new PC);
                pcl::ExtractIndices<PointT> extract_table;
                extract_table.setNegative(false);
                extract_table.setInputCloud(cloud);
                extract_table.setIndices(clusterPtr);
                extract_table.filter(*tmpcld);
                printf("fun2");

                V4f centroid;
                pcl::compute3DCentroid(*tmpcld, centroid);

                //    std::cout << "Cluster size: " <<
                //    cluster_indices[i_cluster].indices.size() << std::endl;
                //        if ( cluster_indices[i_cluster].indices.size() >
                //        cluster_indices[maxClusterIdx].indices.size() )
                if ((centroid.head<3>() - tblcentroid.head<3>()).norm() < minDist) {
                        maxClusterIdx = i_cluster;
                }
        }

        MyHelperFuns::printString("Number of large clusters found: " +
                                  std::to_string(cluster_indices.size()),
                                  printMsg);

        if (cluster_indices.size() == 0) {
                return;
        }

        pcl::PointIndices::Ptr clusterPtr(
                new pcl::PointIndices(cluster_indices[maxClusterIdx]));

        pcl::ExtractIndices<PointT> extract_table;
        extract_table.setNegative(false);
        extract_table.setInputCloud(cloud);
        extract_table.setIndices(clusterPtr);
        extract_table.filter(*cloud);

        //      /********************** Filter out everything else
        //      **********************/

        //      // Filter out everything but the points in the biggest cluster.
        //      extract.setNegative (false);
        //      extract.setInputCloud (cloudSegmented);
        //      extract.setIndices (clusterPtr);
        //      extract.filter (*cloudSegmented);

        //      extractNormals.setNegative (false);
        //      extractNormals.setInputCloud (cloudSegmentedNormals);
        //      extractNormals.setIndices (clusterPtr);
        //      extractNormals.filter (*cloudSegmentedNormals);

        //     Compute a subsampled version!
        //     To make things a bit faster for feature computation!
        //    pcl::VoxelGrid<PointT> sor;
        //    sor.setInputCloud (cloudSegmented);
        //    sor.setLeafSize (0.003f, 0.003f, 0.003f);
        //    sor.filter (*subSampledSegmentedCloud);
        //    MyHelperFuns::printString("PointCloud after subsampling has:
        //    "+std::to_string(subSampledSegmentedCloud->size())+" data
        //    points.",printMsg);

        //        ne.setSearchMethod (tree);
        //        ne.setInputCloud (subSampledSegmentedCloud);
        //        ne.setRadiusSearch (0.01);
        //        ne.compute (*subSampledSegmentedCloudNormals);
        //     PCLHelperFuns::computeCloudNormals(subSampledSegmentedCloud,0.01,subSampledSegmentedCloudNormals);

        //    MyHelperFuns::printString("Points in the largest clusters:
        //    "+std::to_string(cloudSegmented->size()),printMsg);
}

void SceneSegmentation::segmentPointCloudByTable(const V4f &tablePlane) {
        // TMP storage clouds
        PC::Ptr cloud(new PC);

        // Make a deep copy of original point cloud
        pcl::copyPointCloud(*cloudScene, *cloud);

        MyHelperFuns::printString("Starting point cloud segementation", printMsg);
        MyHelperFuns::printString("PointCloud has: " +
                                  MyHelperFuns::toString(cloud->points.size()) +
                                  " data points.",
                                  printMsg);

        /********************** Filter out distant points **********************/
        // z - camera direction
        // x - horizontal
        // y - vertical

        // pcl::PassThrough<PointT> pass;
        // pass.setInputCloud(cloud);
        // pass.setFilterFieldName("z");
        // pass.setFilterLimits(0.3, 1.0);
        // pass.filter(*cloud);

        // pass.setFilterFieldName("x");
        // pass.setFilterLimits(-0.15, 1.20);
        // pass.filter(*cloud);
        //
        // pass.setFilterFieldName("y");
        // pass.setFilterLimits(-0.10, 1.20);
        // pass.filter(*cloud);


        /************ Remove plane inliers and below plane points *************/
        pcl::PointIndices::Ptr belowPlane(new pcl::PointIndices);
        V4f p;
        for (uint idx = 0; idx != cloud->size(); idx++) {
                p = cloud->points[idx].getVector4fMap();
                p[3] = 1;
                double dp = tablePlane.dot(p);

                if (dp < 0.01 || dp > 0.40)
                        belowPlane->indices.push_back(idx);
        }

        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(true);
        extract.setInputCloud(cloud);
        extract.setIndices(belowPlane);
        extract.filter(*cloudSegmented);

        cloudSegmented->is_dense = false;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloudSegmented,*cloudSegmented, indices);
        cloudSegmented->is_dense = true;



        // Generate cuboid filling table to remove points outside table
        PC::Ptr cubePC(new PC);
        PointT pnt;
        // Left end
        pnt.r = 1.0;   pnt.g = 0.;   pnt.b = 0.;
        pnt.x = -0.214071; pnt.y = 0.0977857; pnt.z = 0.925;
        cubePC->push_back(pnt);
        pnt.x += 0.4*tablePlane[0];
        pnt.y += 0.4*tablePlane[1];
        pnt.z += 0.4*tablePlane[2];
        cubePC->push_back(pnt);
        // Right end
        pnt.x = 0.401198; pnt.y = 0.0820895; pnt.z = 1.214;
        cubePC->push_back(pnt);
        pnt.x += 0.4*tablePlane[0];
        pnt.y += 0.4*tablePlane[1];
        pnt.z += 0.4*tablePlane[2];
        cubePC->push_back(pnt);
        // Left start
        // 0.0106;0.232695;0.53
        pnt.x = 0.0262286; pnt.y = 0.230914; pnt.z = 0.54;
        cubePC->push_back(pnt);
        pnt.x += 0.4*tablePlane[0];
        pnt.y += 0.4*tablePlane[1];
        pnt.z += 0.4*tablePlane[2];
        cubePC->push_back(pnt);
        // Right Start
        // 0.8582638 , -0.0392645 ,  0.51170455
        pnt.x = 0.0262286; pnt.y = 0.230914; pnt.z = 0.54;
        pnt.x += 0.5*(0.8582638); pnt.y += 0.5*(-0.0392645); pnt.z += 0.5*(0.51170455);
        cubePC->push_back(pnt);
        pnt.x += 0.4*tablePlane[0];
        pnt.y += 0.4*tablePlane[1];
        pnt.z += 0.4*tablePlane[2];
        cubePC->push_back(pnt);

        cuboid tableBB;
        PCLHelperFuns::computePointCloudBoundingBox(cubePC,tableBB);

        // Remove points outside cube
        pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
        PCLHelperFuns::selectPointsInCube(cloudSegmented, tableBB, objIndices );

        // Remove object from grasp scene
        extract.setNegative(false);
        extract.setInputCloud(cloudSegmented);
        extract.setIndices(objIndices);
        extract.filter(*cloudSegmented);



















// Remove outliers of the plane some remaining points of the plane.
        pcl::StatisticalOutlierRemoval<PointT> outlierRemove;
        outlierRemove.setInputCloud(cloudSegmented);
        outlierRemove.setMeanK(50);
        outlierRemove.setStddevMulThresh(1.0);
        outlierRemove.filter(*cloudSegmented);

        // PCLHelperFuns::smoothCloud(cloudSegmented);
        // PCLHelperFuns::computeCloudNormals(cloudSegmented, 0.03,
        //                                    cloudSegmentedNormals);





        PCLHelperFuns::smoothCloud3(cloudSegmented,cloudSegmentedNormals);



        // PCLHelperFuns::computeCloudNormals(cloudSegmented, 0.01,
        //                                    cloudSegmentedNormals);

        pcl::VoxelGrid<PointT> vgss;
        vgss.setInputCloud(cloudSegmented);
        vgss.setLeafSize(0.003f, 0.003f, 0.003f);
        vgss.filter(*subSampledSegmentedCloud);
        subSampledSegmentedCloud->is_dense = false;
        // std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*subSampledSegmentedCloud,*subSampledSegmentedCloud, indices);
        subSampledSegmentedCloud->is_dense = true;
        PCLHelperFuns::computeCloudNormals(subSampledSegmentedCloud, 0.01,
                                           subSampledSegmentedCloudNormals);
}

/* Takes the segmented object and extracts a set of images of it. All other
 * pixels in the image are set to black.
 */
void SceneSegmentation::segmentImage() {
        roiObjImgIndices.clear(); // Make sure empty.
        // Extract indices for a window that contains the clustered object.
        int colMin = 1000, rowMin = 1000;
        int colMax = 0, rowMax = 0;
        //  double focalInv = 1000.0/525.0;
        //  int imageCenterX = 640/2;
        //  int imageCenterY = 480/2;

        // Copy pixels for the object into new image and add
        // create a vector where each point is the pixel position of a point cloud
        // point.
        std::vector<int> pxPos(2);
        for (uint idx = 0; idx != cloudSegmented->size(); idx++) {
                //    PointT p = cloudSegmented->points[iter];
                //    // Compute approximate coordinates of pixel position of segmented
                //    object
                //    int col = (1000.0 * p.x / (p.z * focalInv)) + imageCenterX;
                //    int row = (1000.0 * p.y / (p.z * focalInv)) + imageCenterY;
                PCLHelperFuns::pt2px(640, cloudSegmented->points[idx].ID, pxPos);
                int row = pxPos[0];
                int col = pxPos[1];

                // Copy pixels for object
                imgSegmented.at<cv::Vec3b>(row, col) = img.at<cv::Vec3b>(row, col);
                // imgSegmented.at<cv::Vec3b>(row,col)[0] =
                // 0.0;//(cloudSegmentedNormals->points[idx].normal_x)*255.;
                // imgSegmented.at<cv::Vec3b>(row,col)[1] =
                // 0.0;//(cloudSegmentedNormals->points[idx].normal_y)*255.;
                // imgSegmented.at<cv::Vec3b>(row,col)[2] =
                // (cloudSegmentedNormals->points[idx].normal_z)*255.;
                // double d = std::sqrt( std::pow(cloudSegmented->points[idx].x,2) +
                // std::pow(cloudSegmented->points[idx].y,2) +
                // std::pow(cloudSegmented->points[idx].z,2) );
                // imgSegmented.at<cv::Vec3b>(row,col)[2] = d*255.;

                // std::cout << imgSegmented.at<cv::Vec3b>(row,col) << " ";

                // Store row,col positions of the object
                //    std::vector<int> pos(2,0);
                //    pos[0]=row; pos[1]=col;
                roiObjImgIndices.push_back(pxPos);

                // Store the indices in original point cloud for the segmented object
                //    int idx = img.cols * row + col;
                //    objIndices->indices.push_back(idx);

                // Get hold of max and mins positions of the object in the 2D
                // representation.
                if (col < colMin)
                        colMin = col;
                if (col >= colMax)
                        colMax = col;
                if (row < rowMin)
                        rowMin = row;
                if (row >= rowMax)
                        rowMax = row;
        }

        // cv::normalize(imgSegmented, imgSegmented, 0, 255, CV_8UC3);

        // Check that segmented region of the image is not outside boundaries
        if (colMin < 0)
                colMin = 0;
        if (rowMin < 0)
                rowMin = 0;
        if (colMax > 640)
                colMax = 640 - 1;
        if (rowMax > 480)
                rowMax = 480 - 1;

        // Extract rect of interest
        //  Check that our enlarged ROI is not outside of the image.
        rx = colMin - 8;
        ry = rowMin - 8;
        rWidth = colMax - colMin + 18;
        rHeight = rowMax - rowMin + 18;
        if (rx < 0)
                rx = 0;
        if (ry < 0)
                ry = 0;
        if ((rx + rWidth) > 639)
                rWidth = 639 - rx;
        if ((ry + rHeight) > 479)
                rHeight = 479 - ry;
        cv::Mat r1(imgSegmented, cv::Rect(rx, ry, rWidth, rHeight));
        r1.copyTo(imgROI);

        // Set offset
        offset_ = ry * 640 + rx;

        // Set roi indices for object
        // roiObjImgIndices = objImgIndices; // copy
        for (uint i_point = 0; i_point != roiObjImgIndices.size(); i_point++) {
                roiObjImgIndices[i_point][0] = roiObjImgIndices[i_point][0] - ry; // y
                roiObjImgIndices[i_point][1] = roiObjImgIndices[i_point][1] - rx; // x
        }

        // Check that our window is not outside of the image.
        rx = colMin - 5;
        ry = rowMin - 5;
        rWidth = colMax - colMin + 15;
        rHeight = rowMax - rowMin + 15;
        if (rx < 0)
                rx = 0;
        if (ry < 0)
                ry = 0;
        if ((rx + rWidth) > 639)
                rWidth = 639 - rx;
        if ((ry + rHeight) > 479)
                rHeight = 479 - ry;
        cv::Mat r2(img, cv::Rect(rx, ry, rWidth, rHeight));
        r2.copyTo(imgObj);

        //     cv::imshow("Whole Img",imgSegmented);
        //     cv::waitKey();
        // cv::imwrite((rawfileName + "_seg.png"), imgObj);
        cv::imwrite((rawfileName + ".ppm"), imgROI);
        isScene2DSegemented = true;
}


void SceneSegmentation::deleteTmpImFiles()
{
        if (BoostHelperFuns::fileExist((rawfileName + "_seg.png")))
        {
                if( std::remove( (rawfileName + "_seg.png").c_str() ) != 0 )
                        std::perror( ("Error deleting file " + rawfileName + "_seg.png").c_str() );
        }

        if (BoostHelperFuns::fileExist((rawfileName + ".ppm")))
        {
                if( std::remove( (rawfileName + ".ppm").c_str() ) != 0 )
                        std::perror( ("Error deleting file " + rawfileName + ".ppm").c_str() );
        }
}
