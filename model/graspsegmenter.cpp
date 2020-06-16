#include "graspsegmenter.h"

// std
#include <stdexcept>
#include <string>
#include <iostream>

// PCL
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/icp.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>

#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Mine
#include <myhelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>
#include <saccuboid.h>

GraspSegmenter::GraspSegmenter() :
        gCloudSceneIdxs(new pcl::PointIndices),
        gCloudSegmentIdxs(new pcl::PointIndices),
        gCloudSegmentOrigIdxs(new pcl::PointIndices),
        pcGraspedObjectPart(new PC)
{
}


/* Computes the grasp position on an object by comparing a scene with only the
 * object and one where a human is grasping the object.
 *
 * Input:
 *  objSS   - Scene with object not grasped
 *  graspSS - Scene with object grasped.
 * Output:
 *  gCloudSceneIdxs - Point indices for the grasped part of the object in the scene where the object is not grasped.
 *  gImgIdxs - Image pixels indices for the grasped part of the object in the scene where the object is not grasped.
 *  graspBB - Bounding box for the grasp of the object.
 */
void
GraspSegmenter::computeGraspPoints (const SceneSegmentation &objSS, const SceneSegmentation &graspSS)
{
        if( !objSS.isThisSceneSegmented() )
                throw std::runtime_error("Pre-segmentation of scene needed!");

        /************************ POINT CLOUD DIFFERENTATION ************************/
        /* Filter out everything but the object from the scenes without and with agent.
         * by using the indices we collected from the scene segementation of the scene
         * with just the object.
         *
         * setNegative value
         * true = remove points given by indices.
         * false = remove all points but the indices.
         * *****NEVER FORGET!!*****
         */

        // Filter grasping scence out roughly in z and x directions
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud (graspSS.cloudScene);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.3,1.0);
        pass.filter (*(graspSS.cloudScene));

        pass.setFilterFieldName ("x");
        pass.setFilterLimits (-0.1,40);
        pass.filter (*(graspSS.cloudScene));

        PC::Ptr pcObjectGraspedPart(new PC);
        PC::Ptr pcGraspedObject(new PC);

        // Copy segmented cloud original point indices to indices object
        pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
        PCLHelperFuns::cloud2origInd(objSS.cloudSegmented,objIndices);
        // Copy points of the object in the only object scene and the grasped scene
        PCLHelperFuns::filterByOrigID( objSS.cloudScene, objIndices, pcObjectGraspedPart );
        PCLHelperFuns::filterByOrigID( graspSS.cloudScene, objIndices, pcGraspedObject );

        // Compute cloud diff between the cloud of the object and the cloud of the grasped object
        pcl::PointIndices::Ptr graspObjectDiffIdxs (new pcl::PointIndices);
        PCLHelperFuns::detectPointCloudDiffByOrigID(pcObjectGraspedPart, pcGraspedObject, graspObjectDiffIdxs);

        // Filter out the grasped part
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative (false);
        extract.setInputCloud (pcObjectGraspedPart);
        extract.setIndices (graspObjectDiffIdxs);
        extract.filter (*pcObjectGraspedPart);

        // Filter out noise, that is, random pixels that changed during the grasp recording.
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud (pcObjectGraspedPart);
        sor.setMeanK (50);
        sor.setStddevMulThresh (1.0);
        sor.filter(*pcObjectGraspedPart);

        // Match grasped points to points in the full scene
        PCLHelperFuns::matchIndices2SceneByOrigID( pcObjectGraspedPart, objSS.cloudScene, gCloudSceneIdxs); // Replaced with line below
        //PCLHelperFuns::cloud2origInd(pcObjectGraspedPart,gCloudSceneIdxs);

        // Match grasped points to points in the segmented cloud object.
        PCLHelperFuns::matchIndices2SceneByOrigID(pcObjectGraspedPart,objSS.cloudSegmented,gCloudSegmentIdxs);

        // Compute bounding box for grasp!
        PC::Ptr pcBox(new PC(*pcObjectGraspedPart));
        PCLHelperFuns::computePointCloudBoundingBox(pcBox,graspBB);
        PCLHelperFuns::cloud2origInd(pcObjectGraspedPart, gCloudSegmentOrigIdxs);
}


/* Computes the grasp position on an object by comparing a scene with only the
 * object and one where a human is grasping the object.
 *
 * Input:
 *  objSS   - Scene with object not grasped
 *  graspSS - Scene with object grasped.
 * Output:
 *  gCloudSceneIdxs - Point indices for the grasped part of the object in the scene where the object is not grasped.
 *  gImgIdxs - Image pixels indices for the grasped part of the object in the scene where the object is not grasped.
 *  graspBB - Bounding box for the grasp of the object.
 */
void
GraspSegmenter::computeGraspPointsCube (const SceneSegmentation &objSS, const SceneSegmentation &graspSS)
{
        if( !objSS.isThisSceneSegmented() )
                throw std::runtime_error("Pre-segmentation of scene needed!");

        /************************ POINT CLOUD DIFFERENTATION ************************/
        /* Filter out everything but the object from the scenes without and with agent.
         * by using the indices we collected from the scene segementation of the scene
         * with just the object.
         *
         * setNegative value
         * true = remove points given by indices.
         * false = remove all points but the indices.
         * *****NEVER FORGET!!*****
         */

        // Filter grasping scence out rough in z and x directions
        PC::Ptr gpCloud(new PC);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud (graspSS.cloudScene);

        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.3,1.0);
        pass.filter(*gpCloud);

        pass.setFilterFieldName ("x");
        pass.setFilterLimits (-0.1,40);
        pass.filter(*gpCloud);


        filterCloudByPlane(objSS.plCf,gpCloud);

        PC::Ptr pcObjectGraspedPart(new PC);
        PC::Ptr pcGraspedObject(new PC);

        // Extract segmented cloud original point indices
        pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
        PCLHelperFuns::cloud2origInd(objSS.cloudSegmented,objIndices);

        // Extract only the original ID points from the two scences(object & grasp)
        pcl::ExtractIndices<PointT> extract;
        PCLHelperFuns::filterByOrigID( objSS.cloudScene, objIndices, pcObjectGraspedPart);
        PCLHelperFuns::filterByOrigID( gpCloud, objIndices, pcGraspedObject);

        // Smooth the grasped cloud
        PCLHelperFuns::smoothCloud(pcGraspedObject);

        // Compute cloud diff between the cloud of the object and the cloud of the grasped object
        pcl::PointIndices::Ptr graspObjectDiffIdxs (new pcl::PointIndices);
        PCLHelperFuns::detectPointCloudDiffByOrigID(pcObjectGraspedPart, pcGraspedObject, graspObjectDiffIdxs);

        // Filter out the grasped part
        extract.setNegative (false);
        extract.setInputCloud (pcObjectGraspedPart);
        extract.setIndices (graspObjectDiffIdxs);
        extract.filter (*pcObjectGraspedPart);

        // Filter out noise, that is, random pixels that changed during the grasp recording.
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud (pcObjectGraspedPart);
        sor.setMeanK (50);
        sor.setStddevMulThresh (1.0);
        sor.filter(*pcObjectGraspedPart);

        // Compute bounding box for grasp!
//    PCLHelperFuns::computePointCloudBoundingBox(pcObjectGraspedPart,graspBB);
        PCLHelperFuns::fitCuboid2PointCloudPlane(pcObjectGraspedPart,objSS.plCf.head<3>(),graspBB);

        // Compute points inside the bounding box
        PCLHelperFuns::computePointsInsideBoundingBox( objSS.cloudScene, graspBB, gCloudSceneIdxs );
        PCLHelperFuns::computePointsInsideBoundingBox( objSS.cloudSegmented, graspBB, gCloudSegmentIdxs );

        PCLHelperFuns::cloud2origInd(pcObjectGraspedPart, gCloudSegmentOrigIdxs);



}

void
GraspSegmenter::computeGraspPointsCube2 (const SceneSegmentation &objSS, const SceneSegmentation &graspSS)
{
        if( !objSS.isThisSceneSegmented() )
                throw std::runtime_error("Pre-segmentation of scene needed!");

        /************************ POINT CLOUD DIFFERENTATION ************************/
        /* Filter out everything but the object from the scenes without and with agent.
         * by using the indices we collected from the scene segementation of the scene
         * with just the object.
         *
         * setNegative value
         * true = remove points given by indices.
         * false = remove all points but the indices.
         * *****NEVER FORGET!!*****
         */
        //
        // pcl::ConvexHull<PointT> hull;
        // hull.setInputCloud(objSS.cloudSegmented);
        // hull.setDimension(3);
        // std::vector<pcl::Vertices> polygons;
        // PC::Ptr pcGraspNoObject(new PC);
        // PC::Ptr surface_hull (new PC);
        // hull.reconstruct(*surface_hull, polygons);
        //
        // for(int i = 0; i < polygons.size(); i++)
        //   std::cout << polygons[i] << std::endl;
        //
        // // pcl::PointCloud<pcl::PointXYZ>::Ptr objects (new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::CropHull<PointT> bb_filter;
        //
        // bb_filter.setDim(3);
        // bb_filter.setInputCloud(objSS.cloudSegmented);
        // bb_filter.setHullIndices(polygons);
        // bb_filter.setHullCloud(surface_hull);
        // bb_filter.setCropOutside(false);
        // bb_filter.filter(*pcGraspNoObject);

        // std::cout << objects->size() << std::endl;



         // Get object bounding box and points inside the box in the grasp scene
         cuboid objBB;
         PCLHelperFuns::computePointCloudBoundingBox(objSS.cloudSegmented,objBB);
         objBB.x_min -= 0.0;
         objBB.x_max += 0.0;
         objBB.y_min -= 0.0;
         objBB.y_max += 0.0;
         objBB.z_min -= 0.0;
         objBB.z_max += 0.0;
         pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
         PCLHelperFuns::selectPointsInCube(graspSS.cloudSegmented, objBB, objIndices );

         // Remove object from grasp scene
         pcl::ExtractIndices<PointT> extract;
         PC::Ptr pcGraspNoObject(new PC);
         extract.setNegative(true);
         extract.setInputCloud(graspSS.cloudSegmented);
         extract.setIndices(objIndices);
         extract.filter(*pcGraspNoObject);

         // Filter out noise, that is, random pixels that changed during the grasp recording.
         pcl::StatisticalOutlierRemoval<PointT> sor;
         sor.setInputCloud (pcGraspNoObject);
         sor.setMeanK (50);
         sor.setStddevMulThresh (1.0);
         sor.filter(*pcGraspNoObject);


         objBB.x_min -= 0.05;
         objBB.x_max += 0.05;
         objBB.y_min -= 0.05;
         objBB.y_max += 0.05;
         objBB.z_min -= 0.05;
         objBB.z_max += 0.05;
         pcl::PointIndices::Ptr handIndices(new pcl::PointIndices);
         PCLHelperFuns::selectPointsInCube(pcGraspNoObject, objBB, handIndices );

        // pcl::ExtractIndices<PointT> extract;
         // PC::Ptr pcGraspNoObject(new PC);
         extract.setNegative(false);
         extract.setInputCloud(pcGraspNoObject);
         extract.setIndices(handIndices);
         extract.filter(*pcGraspNoObject);



          // Compute bounding box for grasp
          cuboid handBB;
          PCLHelperFuns::computePointCloudBoundingBox(pcGraspNoObject,graspBB);
          // Find points of the object contained in the grasp bounding box
          pcl::PointIndices::Ptr graspObjIndices(new pcl::PointIndices);
          PCLHelperFuns::selectPointsInCube(objSS.cloudSegmented, graspBB, graspObjIndices );
          // Extract points
          PC::Ptr pcObjectGraspedPart(new PC);
          extract.setNegative(false);
          extract.setInputCloud(objSS.cloudSegmented);
          extract.setIndices(graspObjIndices);
          extract.filter(*pcObjectGraspedPart);

          PCLHelperFuns::fitCuboid2PointCloudPlane(pcObjectGraspedPart,objSS.plCf.head<3>(),graspBB);


          // Compute points inside the bounding box
          PCLHelperFuns::computePointsInsideBoundingBox( objSS.cloudScene, graspBB, gCloudSceneIdxs );
          PCLHelperFuns::computePointsInsideBoundingBox( objSS.cloudSegmented, graspBB, gCloudSegmentIdxs );

          PCLHelperFuns::cloud2origInd(pcObjectGraspedPart, gCloudSegmentOrigIdxs);









//
//
//
//         // Filter grasping scence out rough in z and x directions
//         PC::Ptr gpCloud(new PC);
//         pcl::PassThrough<PointT> pass;
//         pass.setInputCloud (graspSS.cloudScene);
//
//         pass.setFilterFieldName ("z");
//         pass.setFilterLimits (0.3,1.0);
//         pass.filter(*gpCloud);
//
//         pass.setFilterFieldName ("x");
//         pass.setFilterLimits (-0.1,40);
//         pass.filter(*gpCloud);
//
//         filterCloudByPlane(objSS.plCf,gpCloud);
//
//
//         PC::Ptr pcObjectGraspedPart(new PC);
//         PC::Ptr pcGraspedObject(new PC);
//
//         // Extract segmented cloud original point indices
//         pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
//         PCLHelperFuns::cloud2origInd(objSS.cloudSegmented,objIndices);
//
//         // Extract only the original ID points from the two scences(object & grasp)
//         pcl::ExtractIndices<PointT> extract;
//         PCLHelperFuns::filterByOrigID( objSS.cloudScene, objIndices, pcObjectGraspedPart);
//         PCLHelperFuns::filterByOrigID( gpCloud, objIndices, pcGraspedObject);
//
//         // Smooth the grasped cloud
//         PCLHelperFuns::smoothCloud(pcGraspedObject);
//
//         // Compute cloud diff between the cloud of the object and the cloud of the grasped object
//         pcl::PointIndices::Ptr graspObjectDiffIdxs (new pcl::PointIndices);
//         PCLHelperFuns::detectPointCloudDiffByOrigID(pcObjectGraspedPart, pcGraspedObject, graspObjectDiffIdxs);
//
//         // Filter out the grasped part
//         extract.setNegative (false);
//         extract.setInputCloud (pcObjectGraspedPart);
//         extract.setIndices (graspObjectDiffIdxs);
//         extract.filter (*pcObjectGraspedPart);
//
//         // Filter out noise, that is, random pixels that changed during the grasp recording.
//         pcl::StatisticalOutlierRemoval<PointT> sor;
//         sor.setInputCloud (pcObjectGraspedPart);
//         sor.setMeanK (50);
//         sor.setStddevMulThresh (1.0);
//         sor.filter(*pcObjectGraspedPart);
//
//         // Compute bounding box for grasp!
// //    PCLHelperFuns::computePointCloudBoundingBox(pcObjectGraspedPart,graspBB);
//         PCLHelperFuns::fitCuboid2PointCloudPlane(pcObjectGraspedPart,objSS.plCf.head<3>(),graspBB);
//
//         // Compute points inside the bounding box
//         PCLHelperFuns::computePointsInsideBoundingBox( objSS.cloudScene, graspBB, gCloudSceneIdxs );
//         PCLHelperFuns::computePointsInsideBoundingBox( objSS.cloudSegmented, graspBB, gCloudSegmentIdxs );
//
//         PCLHelperFuns::cloud2origInd(pcObjectGraspedPart, gCloudSegmentOrigIdxs);



}

bool
GraspSegmenter::computeFeaturesFromGrasp(
        const SceneSegmentation &objSS,
        const MainAxes &FMA,
        const FeatureOpening &FO,
        const FeatureColorHist &FCH,
        const FeatureColorQuantization &FCQ,
        FeatureFPFHBoW &Ffpfh,
        featureHOG &fHOG,
        const FeaturePoseSurfaceAngle &FPSA,
        std::vector<double> &featureV)
{
        featureV.clear();

        if(gCloudSceneIdxs->indices.size()==0) {
                std::cout << "\033[1;31m -- ZERO GRASPED POINTS!! -- \033[0m" << std::endl;
                return false;
        }

        //  0-0   [1] - Object Volumes
        //  1-3   [3] - RANSAC fit score in percentage of inliers to points.
        //  4-5   [2] - Elongatedness
        //  6-8   [3] - Grasp position in percent on the main axes
        //  9-9   [1] - Angle with respect to up-gravity.
        //  10-10 [1] - Free volume in percent of the object bounding box volume.
        //  11-13 [3] - Opening(Above Or Below Opening,Distance from plane, Angle)

        //  14-28 [15] - Color Quantization Histogram
        //  29-31 [3] - Entropy mean var
        //  32-61 [30] - FPFH BoW
        //  62-82 [21] - Pose Surface Angle


        //  [11] - Gradient 1 Histogram
        //  [11] - Gradient 2 Histogram
        //  [11] - Gradient 3 Histogram
        //  [11] - Brightness Histogram (Intensity histogram)


        /** OBJECT FEATURES **/
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


        // printf("Opening(Above Or Below Opening,Distance from plane, Angle): ");
        // // Opening
        // std::vector<double> posOpeningInfo = FO.computePosRelativeOpening(FMA.approachVector,graspBB.transVec);
        // featureV.insert(featureV.end(),posOpeningInfo.begin(),posOpeningInfo.end());
        // MyHelperFuns::printVector(posOpeningInfo);

        /** LOCAL OBJECT FEATURES **/
        printf("Filters:\n");
        // Filters
        std::vector<double> hist;

        printf("Gradient 1: "); fflush(stdout);
        FCH.histGrad(gCloudSegmentOrigIdxs,1,hist);
        assert(hist.size()==11);
        featureV.insert(featureV.end(),hist.begin(), hist.end());
        MyHelperFuns::printVector(hist);

        printf("Gradient 2: "); fflush(stdout);
        FCH.histGrad(gCloudSegmentOrigIdxs,2,hist);
        assert(hist.size()==11);
        featureV.insert(featureV.end(),hist.begin(), hist.end());
        MyHelperFuns::printVector(hist);

        printf("Gradient 3: "); fflush(stdout);
        FCH.histGrad(gCloudSegmentOrigIdxs,3, hist);
        assert(hist.size()==11);
        featureV.insert(featureV.end(),hist.begin(), hist.end());
        MyHelperFuns::printVector(hist);

        printf("Brightness: "); fflush(stdout);
        FCH.histBrightness(gCloudSegmentOrigIdxs, hist);
        assert(hist.size()==11);
        featureV.insert(featureV.end(),hist.begin(), hist.end());
        MyHelperFuns::printVector(hist);


        printf("Color Quantization: "); fflush(stdout);
        // Color Quantization
        std::vector<double> histColorQuantization = FCQ.computePointHist(gCloudSegmentIdxs);
        // assert(histColorQuantization.size()==15);
        featureV.insert(featureV.end(),histColorQuantization.begin(), histColorQuantization.end());
        MyHelperFuns::printVector(histColorQuantization);


        printf("Entropy, Mean, Var: "); fflush(stdout);
        std::vector<double> entropyMeanVar = FCQ.computeEntropyMeanVar(gCloudSegmentIdxs);
        assert(entropyMeanVar.size()==3);
        featureV.insert(featureV.end(),entropyMeanVar.begin(), entropyMeanVar.end());
        MyHelperFuns::printVector(entropyMeanVar);


        printf("BoW: "); fflush(stdout);
        /* Different Resolutions of the FPFH BoW*/
        std::vector<double> cwhist;

        // Best combination 0.02_0.02_20_20 - 0.02_0.06_40_0 - 0.01_0.05_40_20
        cwhist.clear();
        Ffpfh.SetCodeBook("0.01", "0.05", "40", "20");
        Ffpfh.CptBoWRepresentation();
        Ffpfh.GetPtsBoWHist(gCloudSegmentIdxs,cwhist);
        assert(cwhist.size()==40);
        featureV.insert(featureV.end(),cwhist.begin(), cwhist.end());
        MyHelperFuns::printVector(cwhist);

        cwhist.clear();
        Ffpfh.SetCodeBook("0.02", "0.02", "20", "20");
        Ffpfh.CptBoWRepresentation();
        Ffpfh.GetPtsBoWHist(gCloudSegmentIdxs,cwhist);
        assert(cwhist.size()==20);
        featureV.insert(featureV.end(),cwhist.begin(), cwhist.end());
        MyHelperFuns::printVector(cwhist);

        cwhist.clear();
        Ffpfh.SetCodeBook("0.02", "0.06", "40", "0");
        Ffpfh.CptBoWRepresentation();
        Ffpfh.GetPtsBoWHist(gCloudSegmentIdxs,cwhist);
        assert(cwhist.size()==40);
        featureV.insert(featureV.end(),cwhist.begin(), cwhist.end());
        MyHelperFuns::printVector(cwhist);

        printf("HoG: \n"); fflush(stdout);
        cv::Mat graspIm;
        PCLHelperFuns::ID2Img(gCloudSegmentOrigIdxs, objSS.img, graspIm);
        fHOG.setInputSource(graspIm);
        fHOG.compute();
        fHOG.appendFeature(featureV);
        // MyHelperFuns::printVector(cwhist);


        /** GRASP SPECIFIC FEATURES **/

        V3f pos = -graspBB.axisMat.row(0)*0.02;
        std::vector<double> pcentMainAxis = FMA.computePosRelativeToAxes( graspBB.transVec + pos);
        printf("Percent on MainAxes(%lu): ",pcentMainAxis.size());
        //assert(pcentMainAxis.sum()>1E-6);
        featureV.insert(featureV.end(),pcentMainAxis.begin(),pcentMainAxis.end());
        MyHelperFuns::printVector(pcentMainAxis);


        std::vector<double> angleWithUp = FMA.computeGraspAngleWithUp( FMA.approachVector );
        printf("Angle with respect to Up(%lu): ",angleWithUp.size());
        featureV.insert(featureV.end(),angleWithUp.begin(),angleWithUp.end());
        MyHelperFuns::printVector(angleWithUp);


        std::vector<double> freeVolume = FMA.computeFreeVolume( pcGraspedObjectPart );
        printf("Free volume(%lu): ",freeVolume.size());
        featureV.insert(featureV.end(),freeVolume.begin(),freeVolume.end());
        MyHelperFuns::printVector(freeVolume);


        std::vector<double> fpsaVec = FPSA.compute(FMA.approachVector,gCloudSegmentIdxs);
        printf("PoseSurface(%lu): ",fpsaVec.size()); fflush(stdout);
        featureV.insert(featureV.end(),fpsaVec.begin(), fpsaVec.end());
        MyHelperFuns::printVector(fpsaVec);


        printf("Feature vector: ");
        MyHelperFuns::printVector(featureV);
        std::cout << "Feature vector length: " << featureV.size() << std::endl;
        std::cout << "Features Finished" << std::endl;

        return true;
}



void
GraspSegmenter::pushOnEigenXVec2StdVec(const Eigen::VectorXf &eigVec, std::vector<double> &stdVec)
{
        for(uint i=0; i!=eigVec.size(); i++)
                stdVec.push_back(eigVec(i));
}



void
GraspSegmenter::computeGraspPointsSK (const SceneSegmentation &objSS, SceneSegmentation &graspSS)
{

        // Pre process point cloud
//    pcl::PassThrough<PointT> pass;
//    pass.setInputCloud (graspSS.cloudScene);
//    pass.setFilterFieldName ("z");
//    pass.setFilterLimits (0.3,1.0);
//    pass.filter (*(graspSS.cloudScene));

//    pass.setFilterFieldName ("x");
//    pass.setFilterLimits (-0.1,40);
//    pass.filter (*(graspSS.cloudScene));

//    filterCloudByPlane(objSS.plCf,graspSS.cloudScene);

        std::vector<double> paramvec;
        MyHelperFuns::readVecFromFile("paramfile.txt", paramvec);
        //MyHelperFuns::printVector(paramvec);


        // Detect the skin color in the image
        // Constants for finding range of skin color in YCrCb
        cv::Vec3b min_YCrCb(0,133,77);
        cv::Vec3b max_YCrCb (130,173,127);
        cv::Vec3b min_HSV(45,48,80);
        cv::Vec3b max_HSV (230,255,255);

        cv::imshow("Orig Img",graspSS.img);

        // Find skin color in range
        cv::Mat imgCrCb, maskYCrCb, imgMaskedYCrCb;
        cv::cvtColor(graspSS.img, imgCrCb, cv::COLOR_BGR2YCrCb); //Convert the captured frame from BGR to HSV
        cv::inRange(imgCrCb,min_YCrCb,max_YCrCb,maskYCrCb);
        cv::medianBlur(maskYCrCb,maskYCrCb,5);
        graspSS.img.copyTo(imgMaskedYCrCb, maskYCrCb);
        cv::imshow("YCrCb Filter",imgMaskedYCrCb);

//    cv::Mat imgHSV, maskHSV, imgMaskedHSV;
//    cv::cvtColor(graspSS.img, imgHSV, CV_BGR2HSV);
//    cv::inRange(imgHSV,min_HSV,max_HSV,maskHSV);
//    cv::medianBlur(maskHSV,maskHSV,5);
//    imgMaskedYCrCb.copyTo(imgMaskedHSV, maskHSV);
//    cv::imshow("HSV Filter",imgMaskedHSV);

        cv::Mat maskBGR, imgMaskedBGR;
        //cv::cvtColor(imgMaskedYCrCb, imgHSV, CV_BGR2HSV);
        cv::Vec3b min_BGR(0,79,103);
        cv::Vec3b max_BGR(255,110,122);
        cv::inRange(graspSS.img,min_BGR,max_BGR,maskBGR);
        cv::medianBlur(maskBGR,maskBGR,5);
        imgMaskedYCrCb.copyTo(imgMaskedBGR, 255-maskBGR);
        cv::imshow("RGB Filter",imgMaskedBGR);

        cv::Mat maskBGR2, imgMaskedBGR2;
        cv::Vec3b min_BGR2(paramvec[0],paramvec[1],paramvec[2]);
        cv::Vec3b max_BGR2(paramvec[3],paramvec[4],paramvec[5]);
        cv::inRange(imgMaskedBGR,min_BGR2,max_BGR2,maskBGR2);
        cv::medianBlur(maskBGR2,maskBGR2,5);
        imgMaskedBGR.copyTo(imgMaskedBGR2, 255-maskBGR2);
        cv::imshow("RGB Filter2",imgMaskedBGR2);

        cv::Mat maskBGR3, imgMaskedBGR3;
        cv::Vec3b min_BGR3(paramvec[6],paramvec[7],paramvec[8]);
        cv::Vec3b max_BGR3(paramvec[9],paramvec[10],paramvec[11]);
        cv::inRange(imgMaskedBGR2,min_BGR3,max_BGR3,maskBGR3);
        cv::medianBlur(maskBGR3,maskBGR3,5);
        imgMaskedBGR2.copyTo(imgMaskedBGR3, 255-maskBGR3);
        cv::imshow("RGB Filter3",imgMaskedBGR3);

        std::cout << min_BGR3 << std::endl;
        std::cout << max_BGR3 << std::endl;

        cv::Mat maskBGR4, imgMaskedBGR4;
        cv::Vec3b min_BGR4(paramvec[12],paramvec[13],paramvec[14]);
        cv::Vec3b max_BGR4(paramvec[15],paramvec[16],paramvec[17]);
        cv::inRange(imgMaskedBGR3,min_BGR4,max_BGR4,maskBGR4);
        cv::medianBlur(maskBGR4,maskBGR4,5);
        imgMaskedBGR3.copyTo(imgMaskedBGR4, 255-maskBGR4);
        cv::imshow("RGB Filter4",imgMaskedBGR4);

        std::cout << min_BGR4 << std::endl;
        std::cout << max_BGR4 << std::endl;

//    maskBGR1 = 255-maskBGR2;
//    maskBGR2 = 255-maskBGR2;
//    maskBGR3 = 255-maskBGR3;
//    maskBGR4 = 255-maskBGR4;

        maskYCrCb = maskYCrCb.mul(255-maskBGR);
        maskYCrCb = maskYCrCb.mul(255-maskBGR2);
        maskYCrCb = maskYCrCb.mul(255-maskBGR3);
//    maskYCrCb = maskYCrCb.mul(255-maskBGR4);
//    maskYCrCb = maskYCrCb.mul(maskHSV);
//    cv::Mat newMask,imgNewMask;
//    filterSkinColor(graspSS.img,newMask);
//    graspSS.img.copyTo(imgNewMask, maskYCrCb);
//    cv::imshow("New Filter",imgNewMask);

        cv::waitKey(0);

        //int rx,ry,rWidth,rHeight; // Rect of interest
        // Find points in ROI of that is of skincolor and add them to indices set
        // as original IDs
        pcl::PointIndices::Ptr handIndices(new pcl::PointIndices);
        cv::Mat finMask(480,640, CV_8UC1, cv::Scalar(0));
        for(int row=objSS.ry-20; row!=(objSS.ry+objSS.rHeight+20); row++)
                for(int col=objSS.rx-20; col!=(objSS.rx+objSS.rWidth+20); col++)
                        if(maskYCrCb.at<uchar>(row,col)>0)
                        {
                                // Check distances
                                V3f ptGrasp = PCLHelperFuns::ptAtorigID(graspSS.cloudScene, 640*row+col );
                                V3f ptObj = PCLHelperFuns::ptAtorigID(objSS.cloudScene, 640*row+col );
                                double d = (ptGrasp-ptObj).norm();
                                if( d>0.01 && d < 0.08)
                                {
                                        handIndices->indices.push_back(640*row+col);
                                        finMask.at<uchar>(row,col) = 1;
                                }
                        }

//    cv::Mat finMaskImg;
//    graspSS.img.copyTo(finMaskImg, finMask);
//    cv::imshow("Masked Final",finMaskImg);

//    // Compute bounding box for skin colored points.
//    PCLHelperFuns::filterByOrigID(objSS.cloudSegmented,handIndices, pcGraspedObject );
//    PCLHelperFuns::computePointCloudBoundingBox(pcGraspedObject,graspBB);

//    // Compute points inside bounding box
//    PCLHelperFuns::computePointsInsideBoundingBox(objSS.cloudSegmented,graspBB,gCloudSegmentIdxs);
        // Get Full Scence Indices
//    PC::Ptr tmpPC(new PC);
//    pcl::ExtractIndices<PointT> extract;
//    extract.setNegative (false);
//    extract.setInputCloud (objSS.cloudSegmented);
//    extract.setIndices (gCloudSegmentIdxs);
//    extract.filter (*tmpPC);
//    PCLHelperFuns::matchIndices2SceneByOrigID( tmpPC, objSS.cloudScene, gCloudSceneIdxs);
//    // ReCompute bounding box for grasp!
//    PCLHelperFuns::computePointCloudBoundingBox(tmpPC,graspBB);

        PCLHelperFuns::filterByOrigID(graspSS.cloudScene,handIndices, pcGraspedObjectPart );
        // Compute bounding box for grasp!
        PCLHelperFuns::computePointCloudBoundingBox(pcGraspedObjectPart,graspBB);
        // Compute points inside bounding box
        PCLHelperFuns::computePointsInsideBoundingBox(objSS.cloudSegmented,graspBB,gCloudSegmentIdxs );
        // Get Full Scence Indices
        PC::Ptr tmpPC(new PC);
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative (false);
        extract.setInputCloud (objSS.cloudSegmented);
        extract.setIndices (gCloudSegmentIdxs);
        extract.filter (*tmpPC);
        PCLHelperFuns::matchIndices2SceneByOrigID(tmpPC, objSS.cloudScene, gCloudSceneIdxs);
        // ReCompute bounding box for grasp!
        PCLHelperFuns::computePointCloudBoundingBox(tmpPC,graspBB);

}


void
GraspSegmenter::computeGraspPointsGlove (const SceneSegmentation &objSS, SceneSegmentation &graspSS)
{

        /** FILTER THE GRASPED CLOUD BY DISTANCE **/
        PC::Ptr graspCloud(new PC);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud (graspSS.cloudScene);

        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.3,0.70);
        pass.filter (*(graspCloud));

        pass.setFilterFieldName ("x");
        pass.setFilterLimits (-0.1,1.0);
        pass.filter (*(graspCloud));

        filterCloudByPlane(objSS.plCf,graspCloud);
//    PCLHelperFuns::smoothCloud(0.01,graspCloud);

        std::cout << "Part 1" << std::endl << std::flush;

        /** Create img mask from glove threshold **/
        cv::Mat imgHSV, maskHSV;
        cv::cvtColor(graspSS.img, imgHSV, cv::COLOR_BGR2HSV);
        cv::inRange(imgHSV, cv::Scalar(101, 55, 0), cv::Scalar(119, 206, 255), maskHSV);
        int removeSize=5;
        int fillSize = 5;
        //morphological opening (remove small objects from the foreground)
        cv::erode(maskHSV, maskHSV, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(removeSize, removeSize)) );
        cv::dilate(maskHSV,maskHSV, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(removeSize, removeSize)) );
        //morphological closing (fill small holes in the foreground)
        cv::dilate(maskHSV,maskHSV, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(fillSize, fillSize)) );
        cv::erode(maskHSV,maskHSV, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(fillSize, fillSize)) );
//    cv::imshow("HSV Filter",maskHSV);
//    cv::imshow("Orig Image",graspSS.img);

        std::cout << "Part 2" << std::endl << std::flush;

        // Remove outliers from mask
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(maskHSV, contours, hierarchy, cv::RETR_EXTERNAL,  cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
        std::vector<cv::Moments> ContArea(contours.size());
        for( uint i = 0; i < contours.size(); i++ )
        {
                ContArea[i] = cv::moments(contours[i], false);
        }

        std::cout << "Part 3" << std::endl << std::flush;
        std::vector<std::vector<cv::Point> > contours_poly(contours.size());
        cv::Mat finalMask = cv::Mat::zeros( maskHSV.size(), CV_8UC1 );

        for(uint i = 0; i < contours.size(); i++)
        {
                std::cout << ContArea[i].m00 << std::endl;
                if(ContArea[i].m00 > 1000)
                {
                        cv::Scalar color = cv::Scalar(255);
                        cv::drawContours( finalMask, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
//          cv::fillPoly(finalMask, contours[i], cv::Scalar(255));
                }
        }

        // cv::imshow("Fin mask2",finalMask);

        std::cout << "Part 4" << std::endl << std::flush;


        /** From mask extract point cloud indices **/
        //int rx,ry,rWidth,rHeight; // Rect of interest
        // Find points in ROI of that is of skincolor and add them to indices set
        // as original IDs
        pcl::PointIndices::Ptr handIndices(new pcl::PointIndices);
        cv::Mat finMask(480,640, CV_8UC1, cv::Scalar(0));
        for(int row=objSS.ry-20; row!=(objSS.ry+objSS.rHeight+20); row++)
                for(int col=objSS.rx-20; col!=(objSS.rx+objSS.rWidth+20); col++)
                        if(row >= 0 && row < 480 && col >= 0 && col < 640 )
                        {
                                if(finalMask.at<uchar>(row,col)>1)
                                {
                                        // Check distances
                                        //                V3f ptGrasp = PCLHelperFuns::ptAtorigID(graspCloud, 640*row+col );
                                        //                V3f ptObj = PCLHelperFuns::ptAtorigID(objSS.cloudSegmented, 640*row+col );
                                        //                if( ptGrasp.norm()>0 && ptObj.norm()>0 )
                                        {
                                                //                  double d = (ptGrasp-ptObj).norm();
                                                //                  if( d>0.01 && d < 0.08)
                                                {
                                                        handIndices->indices.push_back(640*row+col);
                                                        finMask.at<uchar>(row,col) = 255;
                                                }
                                        }
                                }
                        }

        cv::Mat finMaskImg;
        graspSS.img.copyTo(finMaskImg, finMask);
//    cv::imshow("Masked Final",finMaskImg);
        std::cout << "Part 5" << std::endl << std::flush;
        /** Get point cloud of the hand **/
        PC::Ptr gc(new PC);
        PCLHelperFuns::filterByOrigID(graspCloud,handIndices,gc );
        pcl::StatisticalOutlierRemoval<PointT> outlierRemove;
        outlierRemove.setInputCloud (gc);
        outlierRemove.setMeanK (10);
        outlierRemove.setStddevMulThresh (1.0);
        outlierRemove.filter (*gc);

        // Compute bounding box for grasp!
        //PCLHelperFuns::computePointCloudBoundingBox(pcGraspedObject,graspBB);
        PCLHelperFuns::fitCuboid2PointCloudPlane(gc,objSS.plCf.head<3>(),graspBB);

        graspBB.transVec(2) += 0.02;

        // Compute points inside bounding box
        PCLHelperFuns::computePointsInsideBoundingBox(objSS.cloudSegmented,graspBB,gCloudSegmentIdxs );
        if(gCloudSegmentIdxs->indices.size()<100)
        {
                graspBB.x_min -= 0.01;
                graspBB.x_max += 0.01;
                graspBB.y_min -= 0.01;
                graspBB.y_max += 0.01;
                graspBB.z_min -= 0.01;
                graspBB.z_max += 0.04;
                graspBB.depth += 0.02;
                graspBB.width += 0.02;
                graspBB.height += 0.02;
                PCLHelperFuns::computePointsInsideBoundingBox(objSS.cloudSegmented,graspBB,gCloudSegmentIdxs);
        }

        std::cout << "Part 6" << std::endl << std::flush;
        // Get Full Scence Indices
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative (false);
        extract.setInputCloud (objSS.cloudSegmented);
        extract.setIndices (gCloudSegmentIdxs);
        extract.filter (*pcGraspedObjectPart);
        PCLHelperFuns::matchIndices2SceneByOrigID(pcGraspedObjectPart, objSS.cloudScene, gCloudSceneIdxs);
        PCLHelperFuns::cloud2origInd(pcGraspedObjectPart, gCloudSegmentOrigIdxs);
}


void
GraspSegmenter::detectPointCloudDiffByOrigID(const PC::Ptr &cloud1, const PC::Ptr &cloud2, pcl::PointIndices::Ptr &indices)
{
        std::vector<double> paramvec;
        MyHelperFuns::readVecFromFile("pcdiffparam.txt", paramvec);

        PC::const_iterator pc1Iter;
        PC::const_iterator pc2Iter;
        int ptIdx = 0;
        bool ptFound = false;
        for(pc1Iter=cloud1->begin(); pc1Iter!=cloud1->end(); ++pc1Iter)
        {
                ptFound = false;
                for(pc2Iter=cloud2->begin(); pc2Iter!=cloud2->end(); ++pc2Iter)
                {
                        if(pc1Iter->ID == pc2Iter->ID)
                        {
                                ptFound = true;
                                break;
                        }

                }

                if(ptFound)
                {
                        //printf("Point found.\n");
                        V3f vecDiff = pc1Iter->getVector3fMap() - pc2Iter->getVector3fMap();
                        //V3f colorDiff(pc1Iter->r-pc2Iter->r,pc1Iter->g-pc2Iter->g,pc1Iter->b-pc2Iter->b);
                        cv::Mat c1(1,1, CV_8UC3, cv::Scalar(pc1Iter->r,pc1Iter->g,pc1Iter->b));
                        cv::Mat c2(1,1, CV_8UC3, cv::Scalar(pc2Iter->r,pc2Iter->g,pc2Iter->b));
                        cv::cvtColor(c1, c1, cv::COLOR_RGB2Lab);
                        cv::cvtColor(c2, c2, cv::COLOR_RGB2Lab);
                        cv::Vec3b c1Val = c1.at<cv::Vec3b>(0,0);
                        cv::Vec3b c2Val = c2.at<cv::Vec3b>(0,0);
                        float colorDist = cv::norm(c1Val-c2Val);
                        //std::cout << "Distance " << colorDist << std::endl;
                        double d = vecDiff.norm();
                        if( d>paramvec[0] && d<paramvec[1] )
                        { indices->indices.push_back(ptIdx); }
                }
                ptIdx++;
        }
}




void
GraspSegmenter::filterSkinColor(const cv::Mat &img,cv::Mat &mask)
{

        mask = cv::Mat::zeros(480,640,CV_8UC1);

        cv::Mat imHSV;
        img.convertTo(imHSV, CV_32FC3,1.0/255.0);
        cv::cvtColor(imHSV, imHSV, cv::COLOR_BGR2HSV);

        for(int row=0; row!=480; row++)
        {
                for(int col=0; col!=640; col++)
                {
                        cv::Vec3f c = imHSV.at<cv::Vec3f>(row,col);
                        float H = c(0); float S = c(1); float V = c(2);
                        // Convert
                        H=H-180.0; S=100.0*S; V=100.0*V;
                        //std::cout << H <<","<< S << "," << V << std::endl;
                        bool cond1 = S>=10.0;
                        bool cond2 = V>=40.0;
                        bool cond3 = S<=(-H-0.1*V+110.0);
                        bool cond4 = H<=(-0.4*V+75.0);
                        bool cond5 = ( H>=0.0 && S<=(0.08*(100.0-V)*H+0.5*V) ) || (H<0.0 && S<=(0.5*H+35.0));
                        if(cond1 && cond2 && cond3 && cond4 && cond5)
                                mask.at<uchar>(row,col) = 255;
                }
        }

}


void
GraspSegmenter::filterCloudByPlane(const V4f &plCf,PC::Ptr &cloud)
{
        pcl::PointIndices::Ptr belowPlane (new pcl::PointIndices);
        for( uint idx = 0; idx != cloud->size(); idx++ )
        {
                V4f p = cloud->points[idx].getVector4fMap();
                p[3] = 1;
                if( plCf.dot(p) < 0.005 ) belowPlane->indices.push_back(idx);
        }

        pcl::ExtractIndices<PointT> extract;
        extract.setNegative (true);
        extract.setInputCloud (cloud);
        extract.setIndices (belowPlane);
        extract.filter (*cloud);
}
