//STD
#include <iostream>
// PCL
#include "PCLTypedefs.h"
#include <pcl/io/pcd_io.h>
// Opencv
#include <opencv2/highgui.hpp>
// Mine
//#include "myhelperfuns.h"
#include "boosthelperfuns.h"
#include "scenesegmentation.h"

int
main(int argc, char **argv)
{

    /** Set path names. **/
    /* The scence directory contains the
   * pcd files containing recordings of scenes with just
   * one object present. The scene_grasp directory contains
   * files with the name foo1.pcd, foo2.pcd and so on. Where
   * foo referes to a scene file in the scene directory.
   */
    if(argc < 2)
    {
        std::cout << "Specify path to PCD files! (With trailing slash)" << std::endl;
        return 0;
    }

    if(argc < 3)
    {
        std::cout << "Specify path where you want to save the PCD files! (With trailing slash)" << std::endl;
        return 0;
    }


    boost::filesystem::path sourceDirName ((std::string(argv[1])));
    if(!boost::filesystem::is_directory(sourceDirName))
    {
        std::cout << "Source directory does not exist!" << std::endl;
        return -1;
    }

    std::string dirOutput = argv[2];
    boost::filesystem::path outputDirName (dirOutput);
    if(!boost::filesystem::is_directory(outputDirName))
    {
        if (boost::filesystem::create_directory(outputDirName))
        {
            std::cout << "Directory not found. But now its created!" << std::endl;
        }
    }


    // 1. Find all files in source directory
    std::vector<boost::filesystem::path> fileNames;
    BoostHelperFuns::getListOfFilesInDir(sourceDirName, ".pcd", fileNames);


    for(std::vector<boost::filesystem::path>::iterator sceneFileNamePtr = fileNames.begin(); sceneFileNamePtr!=fileNames.end(); ++sceneFileNamePtr)
    {

        std::string sceneFileFullPath = sourceDirName.string() + sceneFileNamePtr->string();
        std::cout << std::endl << sceneFileFullPath << std::endl;
        //MyHelperFuns::writeStringToFile("fileNamesScene.txt",sceneFileFullPath);

        /************************ SCENE SEGMENTATION ************************/
        printf("Segmenting 3D scene..\n");
        // Do 3D segmentation
        SceneSegmentation SS;
        SS.setInputSource(sceneFileFullPath);
        // Segmentation of 3D Scene
        SS.segmentPointCloud();
        // SS.rmHalfPC(0);
        if(SS.cloudSegmented->size()<200)
        {
            std::cout << "Found no object in scene. Continuing with next file.." << std::endl;
            continue;
        }

        SS.segmentImage();

        // Write PCD to file
        int lastindex = sceneFileNamePtr->string().find_last_of(".");
        std::string rawfileName = sceneFileNamePtr->string().substr(0, lastindex);

        std::cout << "Saving " << dirOutput+rawfileName+"_segmented.pcd" << std::endl;
        pcl::io::savePCDFile (dirOutput+rawfileName+"_segmented.pcd", *SS.cloudSegmented, true);

        // Write img to file
        cv::imwrite(dirOutput+rawfileName+"_segmented.png",SS.imgObj);

    }

    return 1;
}
