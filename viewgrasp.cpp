// std
#include <string>
#include <stdexcept>
#include <ctime>
// PCL
#include "PCLTypedefs.h"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>

#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Mine
#include <myhelperfuns.h>
#include "cuboid.hpp"
#include <pclhelperfuns.h>
#include <boosthelperfuns.h>
#include <eigenhelperfuns.h>
#include <mainaxes.h>
#include <featurecolorquantization.h>
#include <featureopening.h>
#include <graspsegmenter.h>
#include <scenesegmentation.h>
#include <symmetrycompletion.h>

void colorIndices(PC::Ptr &cloud, const pcl::PointIndices::Ptr &indPtr, const V3f &rgb);
void colorCloud(PC::Ptr &cloud,const V3f &rgb);

/* MAIN */
int
main(int argc, char **argv)
{
        if(argc < 3)
        {
                std::cout << "Arguments: [1] Object on table, [2] Grasp of object." << std::endl;
                return 0;
        }

        std::string fName1 = argv[1];
        std::string fName2 = argv[2];

        if(!BoostHelperFuns::fileExist(fName1) || !BoostHelperFuns::fileExist(fName2))
        {
                std::cout << "File not found!" << std::endl;
                return 0;
        }


        /** SEGMENTATION **/
        // For agent empty scene segment out object.
        SceneSegmentation SS_obj;
        SS_obj.setInputSource(fName1);
        SS_obj.segmentPointCloud();
        SS_obj.segmentImage();

        // /** FEATURES **/
        MainAxes FMA;
        FMA.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals, SS_obj.plCf.head<3>());
        FMA.fitObject2Primitives();

        // // Detect Opening
        // // printf("Detecting openings..\n");
        // FeatureOpening FO;
        // FO.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals, FMA, SS_obj.imgROI);
        // // FO.detectOpening();

        // // Quantize colors
        // printf("Quantizing colors..\n");
        // FeatureColorQuantization FCQ;
        // FCQ.setInputSource(SS_obj.rawfileName,SS_obj.imgROI,SS_obj.roiObjImgIndices,SS_obj.offset_,SS_obj.cloudSegmented);
        // FCQ.colorQuantize();

        // // Sobel, Laplacian and Itensity
        // printf("Applying Gradient and Itensity filters..\n");
        // FeatureColorHist FCH;
        // FCH.setInputSource(SS_obj.imgSegmented);
        // FCH.computeFeatureMats();

        // // FPFH
        // printf("Computing FPFH BoW representation..\n");
        // featureFPFH Ffpfh;
        // Ffpfh.setInputSource(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals);
        // // Ffpfh.setCodeBook("0.02","0.06", "40", "20");
        // // Ffpfh.cptBoWRepresentation();

        // printf("HoG: "); fflush(stdout);
        // featureHOG fHOG;
        // fHOG.setInputSource(SS_obj.img);
        // fHOG.compute();



        // // PoseSurfaceAngle
        // FeaturePoseSurfaceAngle FPSA;
        // FPSA.setInputSource(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals);


        // Smmetry completion
        //    std::clock_t startTime; // Timer
        //    startTime = std::clock();
        //    SymmetryCompletion SC;
        //    V3f vp(0,0,1);
        //    vp.normalize();
        //    SC.setInputSources(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals,vp,FMA);
        //    SC.completeCloud();
        //    std::cout << "Size of new point cloud: " << SS_obj.cloudSegmented->size() << std::endl;
        //    std::cout << "Symmetry computation time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;

           // FMA.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals,SS_obj.plCf.head<3>());
           // FMA.fitObject2Primitives();


        // Generate cuboid filling table
        PC::Ptr cubePC(new PC);
        PointT pnt;
        // Left end
        pnt.r = 1.0;   pnt.g = 0.;   pnt.b = 0.;
        pnt.x = -0.214071; pnt.y = 0.0977857; pnt.z = 0.925;
        cubePC->push_back(pnt);
        pnt.x += 0.35*SS_obj.plCf[0];
        pnt.y += 0.35*SS_obj.plCf[1];
        pnt.z += 0.35*SS_obj.plCf[2];
        cubePC->push_back(pnt);
        // Right end
        pnt.x = 0.401198; pnt.y = 0.0820895; pnt.z = 1.214;
        cubePC->push_back(pnt);
        pnt.x += 0.35*SS_obj.plCf[0];
        pnt.y += 0.35*SS_obj.plCf[1];
        pnt.z += 0.35*SS_obj.plCf[2];
        cubePC->push_back(pnt);
        // Left start
        // 0.0106;0.232695;0.53
        pnt.x = 0.0262286; pnt.y = 0.230914; pnt.z = 0.54;
        cubePC->push_back(pnt);
        pnt.x += 0.35*SS_obj.plCf[0];
        pnt.y += 0.35*SS_obj.plCf[1];
        pnt.z += 0.35*SS_obj.plCf[2];
        cubePC->push_back(pnt);
        // Right Start
    // 0.8582638 , -0.0392645 ,  0.51170455
        pnt.x = 0.0262286; pnt.y = 0.230914; pnt.z = 0.54;
        pnt.x += 0.5*(0.8582638); pnt.y += 0.5*(-0.0392645); pnt.z += 0.5*(0.51170455);
        cubePC->push_back(pnt);
        pnt.x += 0.35*SS_obj.plCf[0];
        pnt.y += 0.35*SS_obj.plCf[1];
        pnt.z += 0.35*SS_obj.plCf[2];
        cubePC->push_back(pnt);

        cuboid tableBB;
        PCLHelperFuns::computePointCloudBoundingBox(cubePC,tableBB);




        /** GRASP BOUNDING BOX **/
        // Read agent grasping object scene
        SceneSegmentation SS_grasp;
        SS_grasp.setInputSource(fName2);
        // Do fake segmentation to find the approach vector
        SS_grasp.segmentPointCloudByTable(SS_obj.plCf);
        FMA.findApproachVector(SS_obj.cloudSegmented,SS_grasp.cloudSegmented);


        // Create grasp computer and compute the grasp from both positions
        GraspSegmenter GS;
        //GS.computeGraspPoints(SS_obj,SS_grasp);
        //GS.computeGraspPoints(SS_obj,SS_grasp);
        //    GS.computeGraspPointsCube(SS_obj,SS_grasp);
        GS.computeGraspPointsCube2(SS_obj,SS_grasp);
        V3f posOut = -GS.graspBB.axisMat.row(0)*0.02;
        MyHelperFuns::printVector(FMA.computePosRelativeToAxes(GS.graspBB.transVec + posOut ),"Grasp position relative to main axis");
        MyHelperFuns::printVector(FMA.computeFreeVolume(GS.pcGraspedObjectPart),"Volume: ");
        printf("Angle with respect to the table: %f",RAD2DEG(FMA.computeGraspAngleWithUp(FMA.approachVector)[0]));

        std::vector<double> featureVec;
        // GS.computeFeaturesFromGrasp(FMA, FO, FCH, FCQ, Ffpfh, FPSA,featureVec);
        // GS.computeFeaturesFromGrasp(SS_obj, FMA, FO, FCH, FCQ, Ffpfh, fHOG, FPSA, featureVec);


        /** COLOR GRASPED PARTS **/
        //    PCLHelperFuns::colorIndices(SS_obj.cloudScene,GS.gCloudSceneIdxs,V3f(255,0,0));
        PCLHelperFuns::colorIndices(SS_obj.cloudSegmented,GS.gCloudSegmentIdxs,V3f(255,0,0));
        std::cout << SS_obj.cloudSegmented->size() << " "  << GS.gCloudSegmentIdxs->indices.size() << std::endl;
        //    PC::Ptr cubeCloud(new PC);
        //    pcl::ExtractIndices<PointT> extract;
        //    extract.setNegative (false);
        //    extract.setInputCloud (SS_obj.cloudSegmented);
        //    extract.setIndices(GS.gCloudSegmentIdxs);
        //    extract.filter(*cubeCloud);


        /** VIEWS **/
        // View the differing parts
        int v1(0), v2(1), v3(2), v4(3);
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (1, 1, 1);
        viewer->initCameraParameters ();
        viewer->setCameraPosition(0,0,0,0,-1,1,0);

        // PC::Ptr handCloud(new PC);
        // pcl::PointIndices::Ptr ptIdxs(new pcl::PointIndices);
        // PCLHelperFuns::cloud2origInd(SS_obj.cloudSegmented, ptIdxs);
        // PCLHelperFuns::filterOutByOrigID(ptIdxs, SS_grasp.cloudSegmented, handCloud);
        // PCLHelperFuns::filterCloudFromCloud(SS_grasp.cloudSegmented,SS_obj.cloudSegmented, handCloud,0.02);
        // pcl::StatisticalOutlierRemoval<PointT> outlierRemove;
        // outlierRemove.setInputCloud(handCloud);
        // outlierRemove.setMeanK(200);
        // outlierRemove.setStddevMulThresh(1.0);
        // outlierRemove.filter(*handCloud);

        /** THREE VIEWPOINTS **/
        /*
        // Viewport V1
        viewer->createViewPort (0.0, 0.0, 0.33, 1.0, v1);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(SS_grasp.cloudScene);
        viewer->addPointCloud<PointT> (SS_grasp.cloudScene, rgb1, "Whole scene", v1);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,"Whole scene");
        viewer->setBackgroundColor (.827, .827, .827,v1);

        // Viewport V2
        viewer->createViewPort (0.33, 0.0, 0.66, 1.0, v2);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(SS_obj.cloudSegmented);
        //  viewer->addPointCloud<PointT> (SS_obj.cloudSegmented, rgb3, "Hand", v3);
        viewer->addPointCloud<PointT> (SS_obj.cloudSegmented, rgb2, "Segmented Scence", v2);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,"Segmented Scence");
        viewer->setBackgroundColor (.827, .827, .827,v2);

        // Viewport V3
        colorCloud(GS.pcGraspedObjectPart,V3f(0,255,255));
        viewer->createViewPort (0.66, 0.0, 1.0, 1.0, v3);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(GS.pcGraspedObjectPart);
        //  viewer->addPointCloud<PointT> (SS_obj.cloudSegmented, rgb3, "Hand", v3);
        viewer->addPointCloud<PointT> (GS.pcGraspedObjectPart, rgb3, "Hand", v3);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,"Hand");
        viewer->setBackgroundColor (.827, .827, .827,v3);
        */

        // // Get max min points of object point cloud
        // PointT min_pt, max_pt;
        // pcl::getMinMax3D(*SS_obj.cloudSegmented,min_pt,max_pt);
        //
        // // Filter out object out of grasped cloud
        // pcl::PassThrough<PointT> pass;
        // pass.setInputCloud (SS_grasp.cloudSegmented);
        // pass.setFilterLimitsNegative(true);
        //
        // pass.setFilterFieldName ("x");
        // pass.setFilterLimits (min_pt.x+0.03,max_pt.x-0.03);
        // pass.filter(*SS_grasp.cloudSegmented);
        //
        // pass.setFilterFieldName ("y");
        // pass.setFilterLimits (min_pt.y+0.03,max_pt.y-0.03);
        // pass.filter(*SS_grasp.cloudSegmented);
        //
        // pass.setFilterFieldName ("z");
        // pass.setFilterLimits (min_pt.z+0.03,max_pt.z-0.03);
        // pass.filter(*SS_grasp.cloudSegmented);

        cuboid graspBB;
        PCLHelperFuns::computePointCloudBoundingBox(SS_obj.cloudSegmented,graspBB);
        pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
        PCLHelperFuns::selectPointsInCube( SS_grasp.cloudSegmented, graspBB, objIndices );

        pcl::ExtractIndices<PointT> extract;
        extract.setNegative(true);
        extract.setInputCloud(SS_grasp.cloudSegmented);
        extract.setIndices(objIndices);
        extract.filter(*SS_grasp.cloudSegmented);

        /** TWO VIEWPOINTS **/
        // Viewport V1
        viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb10(SS_grasp.cloudSegmented);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb11(SS_obj.cloudSegmented);
        viewer->addPointCloud<PointT> (SS_grasp.cloudSegmented, rgb10, "Grasp", v1);
        viewer->addPointCloud<PointT> (SS_obj.cloudSegmented, rgb11, "Object", v1);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,"Grasp");
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,"Object");
        viewer->setBackgroundColor (.827, .827, .827,v1);
        viewer->addCube(tableBB.transVec,tableBB.quartVec,tableBB.width,tableBB.height,tableBB.depth,"CubeTable",v1);
        // viewer->addCube(-0.1,1.,-0.1,0.5,0.75,1.,.5,0.5,0.5,"CubeTable",v1);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "CubeTable");

        // Viewport V2
        viewer->createViewPort (0.5, 0.0, 1., 1.0, v2);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(SS_obj.cloudSegmented);
        //  viewer->addPointCloud<PointT> (SS_obj.cloudSegmented, rgb3, "Hand", v3);
        viewer->addPointCloud<PointT> (SS_obj.cloudSegmented, rgb2, "Segmented Scence", v2);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,"Segmented Scence");
        viewer->setBackgroundColor (.827, .827, .827,v2);




        viewer->addCube(GS.graspBB.transVec,GS.graspBB.quartVec,GS.graspBB.width,GS.graspBB.height,GS.graspBB.depth,"CubeBB",v2);
        // viewer->setShapeRenderingProperties ( pcl::visualization::PCL_VISUALIZER_OPACITY, 50, "CubeBB" );
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "CubeBB");


        // Add grasp direction vector
        Eigen::Vector3f pEnd;
        Eigen::Vector3f pStart = FMA.midPoint + FMA.approachVector * FMA.approachVectorLength;
        for(int idx=0; idx!=1; idx++)
        {
                pEnd =  FMA.midPoint; //pStart - FMA.approachVector * FMA.approachVectorLength;
                PointT p1, p2;
                PCLHelperFuns::convEigen2PCL(pEnd,p1);
                PCLHelperFuns::convEigen2PCL(pStart,p2);
                viewer->addArrow(p2,p1, 1.0, 0.0, 0, 0, MyHelperFuns::toString(idx),v1);
        }




        while (!viewer->wasStopped ())
        {
                viewer->spin ();
                boost::this_thread::sleep (boost::posix_time::microseconds (10000));
        }


        SS_obj.deleteTmpImFiles();
        return 0;
}




void
colorIndices(PC::Ptr &cloud, const pcl::PointIndices::Ptr &indPtr, const V3f &rgb)
{
        // Set circle inliers to the color of whatever RGB
        std::vector<int>::iterator iter = indPtr->indices.begin();
        for(; iter!=indPtr->indices.end(); ++iter)
        {
                cloud->points[*iter].r = rgb(0);
                cloud->points[*iter].g = rgb(1);
                cloud->points[*iter].b = rgb(2);
        }
}


void
colorCloud(PC::Ptr &cloud,const V3f &rgb)
{
        for(PC::iterator iter=cloud->begin(); iter!=cloud->end(); ++iter)
        {
                iter->r = rgb(0);
                iter->g = rgb(1);
                iter->b = rgb(2);
        }
}
