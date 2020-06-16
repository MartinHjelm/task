#ifndef SCENESEGMENTATION_H
#define SCENESEGMENTATION_H

// std
#include <string>
#include <vector>

// PCL
#include "PCLTypedefs.h"

// Open CV
#include "opencv2/core.hpp"

class SceneSegmentation
{
    // File name of the PCD file
    std::string fileName;
    bool printMsg;

    // Booleans for remembering if a scene has been segmented or not.
    bool isScene3DSegemented;
    bool isScene2DSegemented;

    int readPointCloudFromFile(const std::string &fName, PC::Ptr &inputCloud);
    void setIDs(PC::Ptr &cloud);

public:

    std::string rawfileName;

    // Point cloud of the scene
    PC::Ptr cloudScene;

    // Point cloud segmented
    PC::Ptr cloudSegmented, subSampledSegmentedCloud;
    PCN::Ptr cloudSegmentedNormals, subSampledSegmentedCloudNormals;

    // Image data
    cv::Mat img; // image of scene
    cv::Mat imgSegmented; // Full image with everything else out blacked but object
    cv::Mat imgROI; // Window of object with everything else blacked but object
    cv::Mat imgObj; // Window of segemented object
    int offset_; //
    int rx,ry,rWidth,rHeight; // Rect of interest

    V4f plCf;

    int pxMin, pxMax, pyMin, pyMax;
    //std::vector<std::vector<int> > objImgIndices; // Indices for pixel position of object in original image
    std::vector<std::vector<int> > roiObjImgIndices; // Indices for pixel position of object in roi image

    // Constructor
    SceneSegmentation();
    ~SceneSegmentation();
    int setInputSource(const std::string &fileName);

    void setupCloud();
    void segmentPointCloud();
    void segmentPointCloudByTable(const V4f &tablePlane);
    void segmentImage();
    void clusterCloud(PC::Ptr &cloud, const V4f &tblcentroid);
    inline bool isThisSceneSegmented() const { if(isScene3DSegemented && isScene2DSegemented) { return true;} else {return false;} }
    int rmHalfPC(const int &partId=0);
    void deleteTmpImFiles();
};

#endif // SCENESEGMENTATION_H
