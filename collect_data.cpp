#include "PCLTypedefs.h"

// std
#include <vector>
#include <string>
#include <stdexcept>

// PCL
#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/pcl_visualizer.h>
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
#include "cuboid.hpp"
#include "scenesegmentation.h"
#include "mainaxes.h"
#include "featurecolorquantization.h"
#include "featurecolorhist.h"
#include "featureopening.h"
// #include "featurefpfh.h"
#include "ftr_fpfh_knn.h"
#include "featureposesurfaceangle.h"
#include "graspsegmenter.h"
#include "symmetrycompletion.h"

int
main(int argc, char **argv)
{



    if(argc<1){ printf("Add filename subfix!\n"); return 0;}

    std::string fnSubfix = argv[1];

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

   std::string dropboxdir = "/Users/martinhjelm";
    // std::string dropboxdir = "/user/martin";


    boost::filesystem::path scencesDirName (dropboxdir + std::string("/Dropbox/Code/DATASETS/pc-datasets/GraspsThesis/scene/"));
    boost::filesystem::path graspsDirName (dropboxdir + std::string("/Dropbox/Code/DATASETS/pc-datasets/GraspsThesis/grasps/"));


    std::string fnFeatureVec = std::string("featureVector_")+fnSubfix+".txt";
    std::string fnFileNamesScence = std::string("fileNamesScene_")+fnSubfix+".txt";
    std::string fnFileNamesGrasps = std::string("fileNamesGrasps_")+fnSubfix+".txt";

    // Remove previous files that we want save to
    std::remove( fnFeatureVec.c_str() );
    std::remove( fnFileNamesScence.c_str() );
    std::remove( fnFileNamesGrasps.c_str() );

    /** FOR EACH OBJECT, O, AND EACH GRASP, G, ON THE OBJECT **/

    // 1. Find all files in non-action scene recordings directory
    std::vector<boost::filesystem::path> sceneFileNames;
    std::vector<std::string> exts = {".pcd"};
    BoostHelperFuns::getListOfFilesInDir(scencesDirName, exts, sceneFileNames);

    std::cout << "Starting feature extraction..." << std::endl;

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



        /************************ FEATURE COMPUTATIONS OVER SEGMENTED OJBECT ************************/
        printf("Computing features over object..\n");

        printf("Computing RANSAC score, primtive and main axes..\n");
        // Compute Main Axes Feature
        MainAxes FMA;
        FMA.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals, SS_obj.plCf.head<3>());
        FMA.fitObject2Primitives();

        // Detect Opening
        printf("Detecting openings..\n");
        FeatureOpening FO;
        FO.setInputSource(SS_obj.cloudSegmented, SS_obj.cloudSegmentedNormals, FMA, SS_obj.imgROI);
        FO.detectOpening();

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
        printf("Computing FPFH BoW representation..\n");
        // featureFPFH Ffpfh;
        FeatureFPFHBoW Ffpfh;
        Ffpfh.SetInputSource(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals);

        // HOG
        printf("Computing HoG BoW representation..\n");
        featureHOG fHOG;
        fHOG.setInputSource(SS_obj.imgObj);
        fHOG.compute();

        // PoseSurfaceAngle
        FeaturePoseSurfaceAngle FPSA;
        FPSA.setInputSource(SS_obj.cloudSegmented,SS_obj.cloudSegmentedNormals);








        printf("COMPUTING GRASPS FOR OBJECT\n");

        /************************ COMPUTE GRASP FEATURE VECTOR ************************/
        // Read grasp files and iterate
        std::vector<boost::filesystem::path> graspFileNames;
        BoostHelperFuns::getListOfFilesInDirWithName(graspsDirName, sceneFileNamePtr->stem().string()+"_",".pcd", graspFileNames);
        if(graspFileNames.size()==0)
        {
            std::cout << "Found no recorded grasps for this object. Continuing with next scene.." << std::endl;
            continue;
        }
        std::vector<double> featureVec;
        for(std::vector<boost::filesystem::path>::iterator graspFileNamePtr = graspFileNames.begin(); graspFileNamePtr!=graspFileNames.end(); ++graspFileNamePtr)
        {
            std::string graspFileFullPath = graspsDirName.string() + graspFileNamePtr->string();
            MyHelperFuns::writeStringToFile(fnFileNamesGrasps,graspFileFullPath);
            //      std::cout << graspFileFullPath << std::endl;

            printf("----------------------------------------------\n");
            printf("\033[92mReading grasp scene of %s\033[0m\n",graspFileNamePtr->c_str());
            SceneSegmentation SS_grasp;
            SS_grasp.setInputSource(graspFileFullPath);
            SS_grasp.segmentPointCloudByTable(SS_obj.plCf);

            // Compute our approach vector
            FMA.findApproachVector(SS_obj.cloudSegmented,SS_grasp.cloudSegmented);

            // Compute grasp feature
            GraspSegmenter GS;
            printf("Computing grasp bounding box and grasp point of contacts..\n");
            GS.computeGraspPointsGlove(SS_obj,SS_grasp);
            //      GS.computeGraspPointsCube(SS,SS_grasp);

            printf("Computing features at grasp..\n");
            if( !GS.computeFeaturesFromGrasp(SS_obj, FMA, FO, FCH, FCQ, Ffpfh, fHOG, FPSA, featureVec) )
                continue;

            printf("Final vector: ");
            MyHelperFuns::printVector(featureVec);

            printf("Writing features to file..\n");
            MyHelperFuns::writeVecToFile(fnFeatureVec,featureVec,";");
        }

        SS_obj.deleteTmpImFiles();
        printf("Finished with scene..starting next..\n");
        printf("*************************************************************\n\n");

    }

    printf("*************************************************************\n\n");
    printf("Done with all files. Over and out.\n");

    return 0;
}
