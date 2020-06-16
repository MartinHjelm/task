// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <time.h>

//  PCL
#include <PCLTypedefs.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/filters/extract_indices.h>
//#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

//#include <stdio.h>
//#include <windows.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


// Mine
#include <scenesegmentation.h>
#include <featurecolorquantization.h>
#include <graspsegmenter.h>
#include <graspfactor.h>
#include <myhelperfuns.h>
#include <boosthelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>
#include <pca.h>
#include <symmetrycompletion.h>

void plotGripper(pcl::visualization::PCLVisualizer::Ptr &viewer, PC::Ptr &cloud, cuboid &graspBB, int itemId, int &viewId, const SceneSegmentation &SS , const MainAxes &MA);

/* MAIN */
int
main(int argc, char **argv)
{

    if(argc < 2)
    {
        std::cout << "Needed arguments: [taskName] [fileName]" << std::endl;
        return 0;
    }

    std::string taskName = argv[1];

    // Read input data
    std::string fName = argv[2];
    if(!BoostHelperFuns::fileExist(fName))
    {
        std::cout << "File not found!" << std::endl;
        return 0;
    }


    int Ngrasps = 200;
    int kN = 3;
    int cldOrPlane = 0;
//    if(argc > 2) Ngrasps = atoi(argv[2]);
//    if(argc > 3) kN = atoi(argv[3]);
//    if(argc > 4) cldOrPlane = atoi(argv[4]);


    // Init viewer and check out point cloud!
    int vc(0), v1(0), v2(0), v3(0), v4(0), v5(0);
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->initCameraParameters ();
    viewer->setCameraPosition(0,0,0,0,-1,1,0);
    viewer->setBackgroundColor (0, 0, 0);
    // Viewport 1
    viewer->createViewPort (0.0, 0.0, 0.16, 1.0, vc);
    viewer->createViewPort (0.16, 0.0, 0.32, 1.0, v1);
    viewer->createViewPort (0.32, 0.0, 0.48, 1.0, v2);
    viewer->createViewPort (0.48, 0.0, 0.64, 1.0, v3);
    viewer->createViewPort (0.64, 0.0, 0.8, 1.0, v4);
    viewer->createViewPort (0.8, 0.0, 1.0, 1.0, v5);
    /************************ SCENE SEGMENTATION ************************/
    SceneSegmentation SS;
    SS.setInputSource(fName);
    // Segmentation of 3D Scene
    SS.segmentPointCloud();
    SS.segmentImage();
    std::cout << "Scene segmentation done.." << std::endl;

    // Compute Main Axes Feature
    MainAxes FMA;
    FMA.setInputSource(SS.cloudSegmented, SS.cloudSegmentedNormals,SS.plCf.head<3>());
    FMA.fitObject2Primitives();

    /************************ SYMMETRY ************************/
    // Do symmetry
    SymmetryCompletion SC;
    V3f vp(0,0,1);
    vp.normalize();
    SC.setInputSources(SS.cloudSegmented,SS.cloudSegmentedNormals,vp,FMA);
    SC.completeCloud();
    FMA.fitObject2Primitives();


    // Copy cloud
    PC::Ptr cloud1(new PC);
    PC::Ptr cloud2(new PC);
    PC::Ptr cloud3(new PC);
    PC::Ptr cloud4(new PC);
    PC::Ptr cloud5(new PC);
    pcl::copyPointCloud(*SS.cloudSegmented,*cloud1);
    pcl::copyPointCloud(*SS.cloudSegmented,*cloud2);
    pcl::copyPointCloud(*SS.cloudSegmented,*cloud3);
    pcl::copyPointCloud(*SS.cloudSegmented,*cloud4);
    pcl::copyPointCloud(*SS.cloudSegmented,*cloud5);


    /************************ GRASP PLANNING ************************/
    printf("Computing features over the object..\n"); fflush(stdout);
    GraspFactor GF(SS);
    GF.loadData(taskName);
    GF.computeFeaturesOverObject();

    //    GF.runGraspPlanner(Ngrasps);
    GF.runGraspPlanner(Ngrasps, kN, cldOrPlane);
    printf("Generated grasps over the object..\n");

    if(GF.graspCuboidsVec.size()==0)
    {
        printf("Found no grasps!");
        return 0;
    }


    /************************ VIEW BEST GRASPS  ************************/

    // Create a point cloud where each point is the center for some grasp and
    // where the color is the measure of probability
    PC::Ptr factorCloud(new PC);
    int j = 0;
    PointT pt;
    std::vector<long unsigned int> sort_indexes = MyHelperFuns::sort_indexes(GF.graspFactorVec);
    for (std::vector<long unsigned int>::const_iterator iter = sort_indexes.begin(); iter!= sort_indexes.end(); ++iter) {
        pt.getVector3fMap() = GF.graspCuboidsVec[*iter].rotVecBack(V3f(0.5*GF.graspCuboidsVec[*iter].width,0,0));
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        factorCloud->push_back(pt);

//        PointT pStart, pEnd;
//        n1 = c.axisMat.row(0);
//        pStart.getVector3fMap() = c.transVec+n1*0.5*c.width;
//        pEnd.getVector3fMap() = pStart.getVector3fMap()+n1*0.05;
//        viewer->addLine(pEnd,pStart,0,1,0,MyHelperFuns::toString(j*1000),vc);

//        if(j>1000) break;

        if(j==0)
            plotGripper(viewer, cloud1, GF.graspCuboidsVec[*iter], 10, v1, SS, FMA );
        else if(j==1)
            plotGripper(viewer, cloud2, GF.graspCuboidsVec[*iter], 20, v2, SS, FMA );
        else if(j==2)
            plotGripper(viewer, cloud3, GF.graspCuboidsVec[*iter], 30, v3, SS, FMA );
        else if(j==3)
            plotGripper(viewer, cloud4, GF.graspCuboidsVec[*iter], 40, v4, SS, FMA );
        else if(j==4)
            plotGripper(viewer, cloud5, GF.graspCuboidsVec[*iter], 50, v5, SS, FMA );


        j++;
    }

    // Plot best gripper cube
    //    int idx = MyHelperFuns::minIdxOfVector(GF.graspFactorVec);
    //plotGripper(viewer, cloud, GF.graspCuboidsVec[idx], 1, v2, SS, FMA );


    std::cout << '\7';
    std::cout << '\a';
    printf("%c", 7);
    /************************ VIEWER ************************/

    // Add clouds
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(SS.cloudSegmented);
    viewer->addPointCloud<PointT> (SS.cloudSegmented, rgb, "Object",vc);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Object");

    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbFC(factorCloud);
    viewer->addPointCloud<PointT> (factorCloud, rgbFC, "Factor Cloud", vc);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Factor Cloud");


    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbCloud1(cloud1);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbCloud2(cloud2);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbCloud3(cloud3);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbCloud4(cloud4);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbCloud5(cloud5);

    viewer->addPointCloud<PointT> (cloud1, rgbCloud1, "GripperCloud1",v1);
    viewer->addPointCloud<PointT> (cloud2, rgbCloud2, "GripperCloud2",v2);
    viewer->addPointCloud<PointT> (cloud3, rgbCloud3, "GripperCloud3",v3);
    viewer->addPointCloud<PointT> (cloud4, rgbCloud4, "GripperCloud4",v4);
    viewer->addPointCloud<PointT> (cloud5, rgbCloud5, "GripperCloud5",v5);





    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (10000));
    }


    return 1;

}



void plotGripper(
        pcl::visualization::PCLVisualizer::Ptr &viewer,
        PC::Ptr &cloud, cuboid &graspBB,
        int itemId, int &viewId,
        const SceneSegmentation &SS,
        const MainAxes &MA)
{
    // Assign some variables we need
    V3f n1 = graspBB.axisMat.col(0);
    V3f n2 = graspBB.axisMat.col(1);
    V3f n3 = graspBB.axisMat.col(2);
    V3f transVec = graspBB.transVec;
    double width = graspBB.width;
    double height = graspBB.height;
    double depth = graspBB.depth;
    graspBB.setAxisRotMat();

    //    graspBB.width = (max_pt.x - min_pt.x);
    //    graspBB.height = 0.05;// * (max_pt.y - min_pt.y);
    //    graspBB.depth = (max_pt.z - min_pt.z);



    //    if(width<height)
    //    {
    //        double tmp = width;
    //        width = height;
    //        height = tmp;
    //    }

    // Add cube to viewer
    viewer->addCube(graspBB.transVec,graspBB.quartVec,graspBB.width,graspBB.height,graspBB.depth,"Cube"+MyHelperFuns::toString(itemId),viewId);

    // 7. Color points inside the
    pcl::PointIndices::Ptr graspedPtsIdxs(new pcl::PointIndices);
    PCLHelperFuns::computePointsInsideBoundingBox(cloud,graspBB,graspedPtsIdxs);
    std::cout<< graspedPtsIdxs->indices.size() << std::endl;
    PCLHelperFuns::colorIndices(cloud,graspedPtsIdxs,V3f(255,0,0));

    // PLOT GRIPPER
    // Approach gripper
    PointT pStart, pEnd;
    pStart.getVector3fMap() = graspBB.rotVecBack(V3f(0.5*width,0,0));
    pEnd.getVector3fMap() = graspBB.rotVecBack(V3f(0.65*width,0,0));
    viewer->addArrow(pStart,pEnd,0,0,1,0,MyHelperFuns::toString(itemId)+MyHelperFuns::toString(0),viewId);

    // Red
    pStart.getVector3fMap() = transVec+n1*0.5*width;
    pEnd.getVector3fMap() = graspBB.rotVecBack(V3f(0.5*width,0,0.5*depth));
    viewer->addLine(pEnd,pStart,1,0,0,MyHelperFuns::toString(itemId)+MyHelperFuns::toString(1),viewId);

//    // Green
    pStart.getVector3fMap() = pEnd.getVector3fMap();
    pEnd.getVector3fMap() = graspBB.rotVecBack(V3f(-0.5*width,0,0.5*depth));
    viewer->addLine(pEnd,pStart,0,1,0,MyHelperFuns::toString(itemId)+MyHelperFuns::toString(2),viewId);

//    // Red
    pStart.getVector3fMap() = transVec+n1*0.5*width;
    pEnd.getVector3fMap() = graspBB.rotVecBack(V3f(0.5*width,0,-0.5*depth));
    viewer->addLine(pEnd,pStart,1,0,0,MyHelperFuns::toString(itemId)+MyHelperFuns::toString(3),viewId);
    // Green
    pStart.getVector3fMap() = pEnd.getVector3fMap();
    pEnd.getVector3fMap() = graspBB.rotVecBack(V3f(-0.5*width,0,-0.5*depth));
    viewer->addLine(pEnd,pStart,0,1,0,MyHelperFuns::toString(itemId)+MyHelperFuns::toString(4),viewId);


    Eigen::MatrixXf trMat(4,4);
    trMat << -0.855453, 0.059819, -0.514415,   0.573975,
            0.506901,  0.300174, -0.808052,   1.22227,
            0.106077, -0.952007, -0.287107,   0.304756,
            0,         0,         0,         1;


    Eigen::MatrixXf trMat2(4,4);
    trMat2 << -0.855453, 0.059819, -0.514415,   0.0,
            0.506901,  0.300174, -0.808052,   0.0,
            0.106077, -0.952007, -0.287107,   0.0,
            0,         0,         0,         1;



    V4f gPoint, approachvector,approachvector2, midPoint;
    gPoint.head<3>() = transVec + n1*0.5*width +0.22*n1; gPoint(3) = 1;

    //    EigenHelperFuns::rotVecDegAroundAxis(n3,60,n1);
    approachvector.head<3>() = -n1.normalized();
    //    approachvector(3) = 0;
    //    approachvector.head<3>() = -SS.plCf.head<3>().normalized();
    approachvector(3) = 1;

    //EigenHelperFuns::rotVecDegAroundAxis(n1,90,n2);
    approachvector2.head<3>() = -n3.normalized();

    //    approachvector(3) = 0;
    //    approachvector.head<3>() = SS.plCf.head<3>().normalized();
    approachvector2(3) = 1;


    midPoint.head<3>() = MA.midPoint;
    midPoint(3) = 1;


    V3f endEfPos = (trMat * gPoint).head<3>();
    V3f aprV1 = (trMat2 * approachvector).head<3>().normalized();
    V3f aprV2 = (trMat2 * approachvector2).head<3>().normalized();
    V3f objCent = (trMat * midPoint).head<3>();


    printf("------- UNROTATED VERSION ----------\n");

    EigenHelperFuns::printEigenVec(endEfPos,"Endeffector position: ");
    //EigenHelperFuns::printMat(approachvector,"Approach position untransformed: ");
    EigenHelperFuns::printEigenVec(aprV1,"Approach Vector: ");
    EigenHelperFuns::printEigenVec(aprV2,"Approach Vector2: ");


    printf("------- ROTATED VERSION ----------\n");

    endEfPos = endEfPos - objCent;
    EigenHelperFuns::rotVecDegAroundAxis(V3f(0,0,1),-180, endEfPos);
    endEfPos = endEfPos + objCent;

    EigenHelperFuns::rotVecDegAroundAxis(V3f(0,0,1),-180, aprV1);
    EigenHelperFuns::rotVecDegAroundAxis(V3f(0,0,1),-180, aprV2);

    EigenHelperFuns::printEigenVec(endEfPos,"Endeffector position: ");
    //EigenHelperFuns::printMat(approachvector,"Approach position untransformed: ");
    EigenHelperFuns::printEigenVec(aprV1,"Approach Vector: ");
    EigenHelperFuns::printEigenVec(aprV2,"Approach Vector2: ");


//    midPoint.head<3>() = SS.plCf.head<3>();
//    midPoint.normalize();
//    midPoint(3) = 1;

//    std::cout << "Plane Normal in robot coordinates" << trMat2 * midPoint << std::endl;



    //PointT pStart, pEnd;
    //    pStart.getVector3fMap() = transVec+SS.plCf.head<3>().normalized();
    //    pEnd.getVector3fMap() = pStart.getVector3fMap()+SS.plCf.head<3>().normalized()*0.75;
    //    viewer->addLine(pStart,pEnd,1,0,0,MyHelperFuns::toString(itemId)+MyHelperFuns::toString(5),viewId);




}



