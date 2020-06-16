#include "PCLTypedefs.h"

// std
#include <vector>
#include <string>
#include <stdexcept>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Mine
#include "myhelperfuns.h"
#include "boosthelperfuns.h"
#include "pclhelperfuns.h"
#include "eigenhelperfuns.h"
#include "cuboid.hpp"
#include "scenesegmentation.h"
#include "mainaxes.h"
#include "featurecolorquantization.h"
#include "featurecolorhist.h"
#include "featureopening.h"
// #include "featurefpfh.h"
#include "ftr_fpfh_knn.h"
#include "featurehog.h"
#include "featureposesurfaceangle.h"
// #include "graspsegmenter.h"
#include "symmetrycompletion.h"

/** EXTRACTS A FULL FEATURE VECTOR FROM THE GIVEN OBJECTS **/

int
main(int argc, char **argv)
{

    if(argc<3){ printf("Usage extract_object_features [1] directory [2] subfix for datafiles \n"); return 0;}

    std::string dirName = argv[1];
    std::string fnSubfix = argv[2];

    /** Set path names. **/
    /* The scence directory contains the
   * pcd files containing recordings of scenes with just
   * one object present. The scene_grasp directory contains
   * files with the name foo1.pcd, foo2.pcd and so on. Where
   * foo referes to a scene file in the scene directory.
   */

    // OLD SYSTEM OF ORDERING FILES
    //    boost::filesystem::path scencesDirName (std::string("/Users/martinhjelm/Dropbox/Code/TaskConstraints/Data/")+fnSubfix+"_transfer/scene/");
    //    boost::filesystem::path graspsDirName (std::string("/Users/martinhjelm/Dropbox/Code/TaskConstraints/Data/")+fnSubfix+"_transfer/scene_grasp/");

//     boost::filesystem::path scencesDirName (std::string("/Users/martinhjelm/Dropbox/Code/DATASETS/pc-datasets/IROS2016/scene/"));
     boost::filesystem::path scencesDirName (dirName);


    std::string fnFeatureVec = std::string("featureVector_")+fnSubfix+".txt";
    std::string fnFileNamesScence = std::string("./fnames ")+fnSubfix+".txt";


    // Remove previous files that we want save to
    std::remove( fnFeatureVec.c_str() );
    std::remove( fnFileNamesScence.c_str() );

    /** FOR EACH OBJECT, O **/

    // 1. Find all files in non-action scene recordings directory
    std::vector<boost::filesystem::path> sceneFileNames;
    std::vector<std::string> exts = {".pcd"};
    BoostHelperFuns::getListOfFilesInDir(scencesDirName, exts, sceneFileNames);


    for(std::vector<boost::filesystem::path>::iterator sceneFileNamePtr = sceneFileNames.begin(); sceneFileNamePtr!=sceneFileNames.end(); ++sceneFileNamePtr)
    {

        std::string sceneFileFullPath = scencesDirName.string() + sceneFileNamePtr->string();
        std::cout << sceneFileFullPath << std::endl;
        MyHelperFuns::writeStringToFile(fnFileNamesScence,sceneFileFullPath);


        /************************ SCENE SEGMENTATION ************************/
        printf("Segmenting 3D scene..\n");
        // Do 3D segmentation
        SceneSegmentation SS_obj;
        SS_obj.setInputSource(sceneFileFullPath);
        // Segmentation of 3D Scene
        SS_obj.segmentPointCloud();
        // SS_obj.rmHalfPC(1);




        if(SS_obj.cloudSegmented->size()<200)
        {
            std::cout << "Found no object in scene. Continuing with next file.." << std::endl;
            continue;
        }
        // Do 2D segmentation
        printf("Segmenting 2D scene..\n");
        SS_obj.segmentImage();

        // Save segmented object
        // pcl::io::savePCDFileBinary("segmentedObjects/segmented_"+sceneFileNamePtr->string(), *(SS_obj.cloudSegmented) );

        // Save ROI of object
        cv::Mat img;
        PCLHelperFuns::cloud2img(SS_obj.cloudSegmented, true, img);
        cv::imwrite(("segmentedObjects/roi_"+sceneFileNamePtr->string()+".png"),img);


        /************************ FEATURE COMPUTATIONS OVER SEGMENTED OJBECT ************************/
        printf("Computing features over object..\n");

        printf("Computing RANSAC score, primtive and main axes..\n");
        // Compute Main Axes Feature
        MainAxes FMA;
        FMA.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals, SS_obj.plCf.head<3>());
        FMA.fitObject2Primitives();

        // Detect Opening
        // printf("Detecting openings..\n");
        // FeatureOpening FO;
        // FO.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals, FMA, SS_obj.imgROI);
        // FO.detectOpening();

        /** SYMMETRI COMPLETION **/
//        SymmetryCompletion SC;
//        V3f vp(0,0,1);
//        vp.normalize();
//        SC.setInputSources(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals,vp,FMA);
//        SC.completeCloud();
//        FMA.fitObject2Primitives();

        // Quantize colors
        printf("Quantizing colors..\n");
        FeatureColorQuantization FCQ;
        FCQ.setInputSource(SS_obj.rawfileName,SS_obj.imgROI,SS_obj.roiObjImgIndices,SS_obj.offset_,SS_obj.cloudSegmented);
        FCQ.colorQuantize();

        // Sobel, Laplacian and Itensity
        printf("Applying Gradient and Itensity filters..\n");
        FeatureColorHist FCH;
        FCH.setInputSource(SS_obj.img);
        FCH.computeFeatureMats();

        // FPFH
        printf("Setting up FPFH BoW representation..\n");
        FeatureFPFHBoW Ffpfh;
        Ffpfh.SetInputSource(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals);

        printf("Computing HoG BoW representation..\n");
        featureHOG fHOG;
        fHOG.setInputSource(SS_obj.imgObj);
        fHOG.compute();


        /************************ COMPUTE OBJECT FEATURE VECTOR ************************/
        printf("Assembling Feature Vector...\n ");

        std::vector<double> featureV;
        pcl::PointIndices::Ptr objectIndices(new pcl::PointIndices);
        pcl::PointIndices::Ptr objectOrigIndices(new pcl::PointIndices);
        for( uint idx = 0; idx != SS_obj.cloudSegmented->size(); idx++ )
        {
            objectIndices->indices.push_back(idx);
        }
        PCLHelperFuns::cloud2origInd(SS_obj.cloudSegmented, objectOrigIndices );

        printf("Object volume: ");
        std::vector<double> volume = FMA.computeObjectVolume();
        featureV.insert(featureV.end(),volume.begin(),volume.end());
        MyHelperFuns::printVector(volume);

        printf("Fitscores: ");
        assert(FMA.fitScores.size()==3);
        featureV.insert(featureV.end(),FMA.fitScores.begin(),FMA.fitScores.end());
        MyHelperFuns::printVector(FMA.fitScores);

        printf("Elongated: ");
        std::vector<double> elonFeature = FMA.computeElongatedness();
        assert(elonFeature.size()==2);
        featureV.insert(featureV.end(),elonFeature.begin(),elonFeature.end());
        MyHelperFuns::printVector(elonFeature);

        printf("Object Dimensions: ");
        std::vector<double> dimFeature = FMA.computeObjectDimensions();
        assert(dimFeature.size()==3);
        featureV.insert(featureV.end(),dimFeature.begin(),dimFeature.end());
        MyHelperFuns::printVector(dimFeature);


        // Opening
// if ( FO.hasOpening)
// {
//     featureV.push_back(1);
//     printf("Opening: Found!\n");
// }
// else
// {
//     featureV.push_back(0);
//     printf("Opening: Not Found.\n");
// }


       printf("Filters:\n");
       // Filters
       std::vector<double> hist;
       FCH.histGrad(objectOrigIndices,1,hist);
       assert(hist.size()==11);
       featureV.insert(featureV.end(),hist.begin(), hist.end());
       MyHelperFuns::printVector(hist,"Grad1");

       FCH.histGrad(objectOrigIndices,2,hist);
       assert(hist.size()==11);
       featureV.insert(featureV.end(),hist.begin(), hist.end());
       MyHelperFuns::printVector(hist,"Grad2");

       FCH.histGrad(objectOrigIndices,3, hist);
       assert(hist.size()==11);
       featureV.insert(featureV.end(),hist.begin(), hist.end());
       MyHelperFuns::printVector(hist,"Grad3:");

        printf("Brightness: ");fflush(stdout);
        FCH.histBrightness(objectOrigIndices, hist);
        assert(hist.size()==11);
        featureV.insert(featureV.end(),hist.begin(), hist.end());
        MyHelperFuns::printVector(hist,"");


        printf("Color Quantization: ");fflush(stdout);
        // Color Quantization
        std::vector<double> histColorQuantization = FCQ.computePointHist2(objectIndices);
        // assert(histColorQuantization.size()==32);
        featureV.insert(featureV.end(),histColorQuantization.begin(), histColorQuantization.end());
        MyHelperFuns::printVector(histColorQuantization,"");


        printf("Entropy, Mean, Var: ");fflush(stdout);
        std::vector<double> entropyMeanVar = FCQ.computeEntropyMeanVar(objectIndices);
        assert(entropyMeanVar.size()==3);
        featureV.insert(featureV.end(),entropyMeanVar.begin(), entropyMeanVar.end());
        MyHelperFuns::printVector(entropyMeanVar,"");


        printf("BoW: \n"); fflush(stdout);
        /* Different Resilutions of the FPFH BoW*/
        std::vector<double> cwhist;

        // Best combination 0.02_0.02_20_20 - 0.02_0.06_40_0 - 0.01_0.05_40_20
        Ffpfh.SetCodeBook("0.01", "0.05", "40", "20");
        Ffpfh.CptBoWRepresentation();
        Ffpfh.GetPtsBoWHist(objectIndices,cwhist);
        assert(cwhist.size()==40);
        featureV.insert(featureV.end(),cwhist.begin(), cwhist.end());
        MyHelperFuns::printVector(cwhist,"BoW 0105");

        cwhist.clear();
        Ffpfh.SetCodeBook("0.02", "0.02", "20", "20");
        Ffpfh.CptBoWRepresentation();
        Ffpfh.GetPtsBoWHist(objectIndices,cwhist);
        assert(cwhist.size()==20);
        featureV.insert(featureV.end(),cwhist.begin(), cwhist.end());
        MyHelperFuns::printVector(cwhist,"BoW 0202");

        cwhist.clear();
        Ffpfh.SetCodeBook("0.02", "0.06", "40", "0");
        Ffpfh.CptBoWRepresentation();
        Ffpfh.GetPtsBoWHist(objectIndices,cwhist);
        assert(cwhist.size()==40);
        featureV.insert(featureV.end(),cwhist.begin(), cwhist.end());
        MyHelperFuns::printVector(cwhist,"BoW 0206");


        fHOG.appendFeature(featureV);

        MyHelperFuns::printVector(featureV,"Feature vector");
        std::cout << "Feature vector length: " << featureV.size() << std::endl;
        std::cout << "Features Finished" << std::endl;


        // printf("Final vector: ");
        // MyHelperFuns::printVector(featureV);

        printf("Writing features to file..\n");
        MyHelperFuns::writeVecToFile(fnFeatureVec,featureV,";");

        SS_obj.deleteTmpImFiles();

        printf("Finished with scene..starting next..\n");
        printf("*************************************************************\n\n");

    }

    printf("*************************************************************\n\n");
    printf("Done with all files. Over and out.\n");

    return 0;
}
