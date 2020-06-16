// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Eigen
#include <Eigen/Dense>

//  PCL
#include <PCLTypedefs.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/filters/extract_indices.h>

// Mine
#include <scenesegmentation.h>
#include <mainaxes.h>
#include <featurecolorquantization.h>
#include <featureopening.h>
#include <featurefpfh.h>
#include <featurecolorhist.h>
#include <featuretexture.h>
#include <myhelperfuns.h>
#include <boosthelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>
#include <symmetrycompletion.h>
#include "featurehog.h"

/* MAIN */
int
main(int argc, char **argv)
{
    if(argc<2)
    {
        std::cout << "Specify file name. Please!" << std::endl;
        std::cout << "Feature Options: axes, primitive, opening, cq, bow[0-5], filter[0-2], hog  " << std::endl;
        return 0;
    }

    std::string fName = argv[1];
    std::string featureName = "";
    std::string sceneFull = "f";
    if(argc > 2) featureName = argv[2];
    if(argc > 4) sceneFull = argv[4];

    if(!BoostHelperFuns::fileExist(fName))
    {
        std::cout << "Point cloud file not found!" << std::endl;
        return 0;
    }

    std::clock_t startTime; // Timer

    /************************ SCENE SEGMENTATION ************************/
    startTime = std::clock();
    SceneSegmentation SS;
    SS.setInputSource(fName);
    // Segmentation of 3D Scene
    if (sceneFull.compare("f")==0)
      SS.segmentPointCloud();
    else
      SS.setupCloud();
    // SS.rmHalfPC(0);
    //Extract point cloud segmentation window into image to run F's 2D segmentation algorithm
    SS.segmentImage();
    std::cout << "Scene segmentation done." << std::endl;
    std::cout << "Scene Segmentation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;


    /************************ FEATURE COMPUTATIONS ************************/
    startTime = std::clock();
    std::cout << "-------------" << std::endl;

    // ## MAINAXES
    MainAxes FMA;
    std::cout << "Computing Main Axes feature" << std::endl;
    // Compute Main Axes Feature
    FMA.setInputSource(SS.cloudSegmented, SS.cloudSegmentedNormals,SS.plCf.head<3>());
    FMA.fitObject2Primitives();
    EigenHelperFuns::printEigenVec(FMA.axesLengths,"Axises lengths");
    std::cout << "Main Axis Computation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;


    // ## OPENING
      FeatureOpening FO;
    PC::Ptr circleCloud(new PC);
    startTime = std::clock();
    std::cout << "-------------" << std::endl;
    std::cout << "Computing Has-Opening feature" << std::endl;
    // Detect Opening
    FO.setInputSource(SS.cloudSegmented, SS.cloudSegmentedNormals,FMA, SS.imgROI);
    FO.detectOpening();
    if(FO.hasOpening)
    {
        std::cout << "Found opening on segmented object" << std::endl;
        //Eigen::Vector3f rgb(255,0,0);
        //PCLHelperFuns::colorIndices(SS.cloudSegmented,FO.circleInliers,rgb);
    }
    else { std::cout << "Found no opening on segmented object" << std::endl; }
    std::cout << "Opening Computation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;


    // ## COLOR QUANTIZATION
    FeatureColorQuantization FCQ;
    startTime = std::clock();
    std::cout << "-------------" << std::endl;
    std::cout << "Computing color quantization feature" << std::endl;
    // Quantize colors
    FCQ.setInputSource(SS.rawfileName,SS.imgROI,SS.roiObjImgIndices,SS.offset_,SS.cloudSegmented);
    FCQ.colorQuantize();
    FCQ.imgCQ2PC();
    // cv::Mat cqImg;
    // FCQ.getColorQuantizedImg(cqImg);
    //        cv::imshow("Segmented object1",SS.imgROI);
    // cv::imshow("Segmented object2",cqImg);
    //        cv::waitKey(0);

    //Colorize object according to quantization
    for(uint iter = 0; iter!=SS.cloudSegmented->size(); iter++)
    {
        V3f cqVals = FCQ.pcCQVals[iter];
        //            EigenHelperFuns::printEigenVec(cqVals,"Color ");
        SS.cloudSegmented->points[iter].r = cqVals(0);
        SS.cloudSegmented->points[iter].g = cqVals(1);
        SS.cloudSegmented->points[iter].b = cqVals(2);
    }
    std::cout << "Color Quantization Computation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;


    // ## FPFH QUANTIZATION
    featureFPFH Ffpfh;
    if(featureName.compare("bow")==0){
        startTime = std::clock();
        std::cout << "-------------" << std::endl;
        printf("Computing FPFH BoW representation..\n");

        // Select parameters for BoW
        std::string radNormals = "0.01";
        std::string radFPFH = "0.01";
        std::string pcaDim = "0";
        std::string cbSize = "0";

        int fpfhidx = 0;
        if(argc > 3) fpfhidx = atoi(argv[3]);


        // Best combination 0.02_0.02_20_20 - 0.02_0.06_40_0 - 0.01_0.05_40_20
        if(fpfhidx==0)
        {
            printf("Picked fpfh features 0 \n");
            radNormals = "0.01";
            radFPFH = "0.05";
            cbSize = "40";
            pcaDim = "20";
        }
        else if(fpfhidx==1)
        {
            printf("Picked fpfh features 1 \n");
            radNormals = "0.02";
            radFPFH = "0.02";
            cbSize = "20";
            pcaDim = "20";
        }
        else if(fpfhidx==2)
        {
            printf("Picked fpfh features 2 \n");
            radNormals = "0.02";
            radFPFH = "0.06";
            cbSize = "40";
            pcaDim = "0";
        }


        Ffpfh.setInputSource(SS.cloudSegmented,SS.cloudSegmentedNormals);
        std::cout << radNormals << radFPFH << cbSize << pcaDim << '\n';
        Ffpfh.setCodeBook(radNormals, radFPFH, cbSize, pcaDim);
        Ffpfh.cptBoWRepresentation();


        // Color object according to BoW code.
        for(uint iVal=0;iVal!=SS.cloudSegmented->size();++iVal)
        {
            std::vector<int> rgb =  MyHelperFuns::getColor(Ffpfh.getBoWForPoint(iVal));
            SS.cloudSegmented->points[iVal].r = rgb[0];
            SS.cloudSegmented->points[iVal].g = rgb[1];
            SS.cloudSegmented->points[iVal].b = rgb[2];
        }
        std::cout << "BoW Computation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
    }

    FeatureColorHist FCH;
    if(featureName.compare("filter")==0)
    {
        startTime = std::clock();
        int filterType = 1;
        if(argc > 3) filterType = atoi(argv[3]);
        std::cout << "Filtertype " << filterType << std::endl;


        FCH.setInputSource(SS.imgSegmented);
        FCH.computeFeatureMats();
        // Color object according to BoW code.
        for(uint iVal=0;iVal!=SS.cloudSegmented->size();++iVal)
        {
            double grad = FCH.gradAtPoint(SS.cloudSegmented->points[iVal].ID,filterType);
            //            std::cout << grad << std::endl;
            SS.cloudSegmented->points[iVal].r = 255*(grad/255.0);
            SS.cloudSegmented->points[iVal].g = 0.;
            SS.cloudSegmented->points[iVal].b = 0.;
        }
        std::cout << "Filter Computation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;
    }

    FeatureTexture FT;
    if(featureName.compare("texture")==0)
    {

        //        cv::Mat imgROI; // Window of object with everything else blacked but object
        //        cv::Mat imgObj; // Window of segemented object
        FT.setInputSource(SS.imgObj);
        std::cout << "Found label " << FT.computeTextureLabel(true) << std::endl;
        return 0;
    }

    featureHOG fHOG;
    if(featureName.compare("hog")==0)
    {
        fHOG.setInputSource(SS.imgObj);
        fHOG.compute();
        return 0;
    }




    if(featureName.compare("opening")==0)
    {
        FO.generateCloudCircle(circleCloud);
    }

    /************************ ADD VIEWER ************************/
    //Add 3 view ports
    int v1(0), v2(0), v3(0);
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    viewer->initCameraParameters ();

    // Viewport 1
    viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(SS.cloudScene);
    viewer->addPointCloud<PointT> (SS.cloudScene, rgb1, "Whole scene",v1);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Whole scene");
    //    viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

    // Viewport 2
    viewer->createViewPort (0.5, 0.0, 1., 1.0, v2);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(SS.cloudSegmented);
    viewer->addPointCloud<PointT> (SS.cloudSegmented, rgb2, "Object",v2);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Object");
    viewer->setBackgroundColor (1., 1., 1.,v2);


    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbC(circleCloud);
    viewer->addPointCloud<PointT> (circleCloud, rgbC, "Circle",v2);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Circle");

    if(featureName.compare("bow")==0)
        viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

    //    // Viewport 3
    //    viewer->createViewPort (0.66, 0.0, 1.0, 1.0, v3);
    ////    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(SS.cloudSegmented);
    ////    viewer->addPointCloud<PointT> (SS.cloudSegmented, rgb3, "SObject",v3);
    ////    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"SObject");

    viewer->setCameraPosition(0,0,0,0,-1,1,0);



    //viewer->addCircle(*FO.pmCircle,"Opening",v2);

    //  ROBOT ORIGIN
    // 566.11077880859375 1227.518310546875 303.9049072265625 -0.18138060909146622 0.20938222550806626 0.76739448403667976 -0.57824377571831764
    //V3f robCoordinate(0.56611077880859375, 0.1227518310546875, 0.03039049072265625);


    //  Add object midpoint axes to viewer
        if(featureName.compare("axes")==0)
        {
            Eigen::Vector3f pEnd;
            Eigen::Vector3f pStart = FMA.midPoint;
            for(uint idx=0; idx!=3; idx++)
            {
                Eigen::Vector3f pTmp = FMA.axesVectors.col(idx);
                pEnd = pStart + pTmp * FMA.axesLengths(idx);
                PointT p1, p2;
                PCLHelperFuns::convEigen2PCL(pStart,p1);
                PCLHelperFuns::convEigen2PCL(pEnd,p2);
                viewer->addArrow(p2,p1, idx*1.33, 1.0, 0, 0, MyHelperFuns::toString(idx),v2);
            }
        }
    //          Add geometric primitives to the viewer
       if(featureName.compare("primitive")==0)
       {

           switch (FMA.objectPrimitive)
           {
           case 0: //"cylinder"
               printf("Adding cylinder\n");
               viewer->addCylinder(*FMA.pmCylinder,"Cylinder",v2);
               break;
           case 1: //"sphere"
               printf("Adding sphere\n");
               viewer->addSphere(*FMA.pmSphere,"Sphere",v2);
               break;
           case 2: //"cuboid"
               printf("Adding cube\n");
               viewer->addCube(FMA.pmCube_.transVec,FMA.pmCube_.quartVec,FMA.pmCube_.width,FMA.pmCube_.height,FMA.pmCube_.depth,"Cube",v2);
               break;
           }
                       viewer->addCube(FMA.pmCube_.transVec,FMA.pmCube_.quartVec,FMA.pmCube_.width,FMA.pmCube_.height,FMA.pmCube_.depth,"Cube",v2);

       }

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (10000));
    }

    return 1;

}
