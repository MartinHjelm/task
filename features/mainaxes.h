#ifndef MAINAXES_H
#define MAINAXES_H

// std
#include <string>

// PCL
#include "PCLTypedefs.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/Vertices.h>

// Mine
#include "cuboid.hpp"


class MainAxes
{
    bool printMsg;
    
    /* Ransac parameters
   * normalDistanceWeight - surface normals influence.
   * distanceThreshold - distance threshold from each inlier point
   * to the model i.e. it cant be great than this
   * radiusLimit - biggest possible radius for cylinder.
   */
    double normalDistanceWeight, distanceThreshold, radiusLimit;

    // Object primitive fitting functions
    double fitCylinder2PointCloud();
    double fitSphere2PointCloud();
    double fitCuboid2PointCloud();
    // Helper function to help limit cylinder and sphere primitives
    void setMaxRadius();
    float cylRadiusMax,sphRadiusMax;

    // Functions for computing features and params from primitives
    void computeFeatureParamsFromCylinder();
    void computeFeatureParamsFromCuboid();
    void computeFeatureParamsFromSphere();
    void computePointCloudBoundingBox();



public:

    typedef boost::shared_ptr<MainAxes > Ptr;
    typedef boost::shared_ptr<const MainAxes > ConstPtr;

    // Constructor
    MainAxes();
    void setInputSource(const PC::Ptr &cPtr, const PCN::Ptr &cloudNormals, V3f tabelNormal);

    // Segmented out point cloud representation of the object we are calculating the feature on.
    PC::Ptr cloud_;
    PCN::Ptr cloudNormals_;

    // Fits a variety of shape primitives to the segemented out point cloud
    void fitObject2Primitives();

    // FEATURE VARIABLES
    Eigen::Matrix3f axesVectors; // col-vectors - width,height,depth
    V3f axesLengths; // width,height,depth
    V3f midPoint;
    int objectPrimitive;
    std::vector<double> fitScores;
    V3f tableNrml;
    V3f approachVector;
    V3f approachVectorStart;
    double approachVectorLength;

    // SHAPE PRIMITIVE PARAMETERS
    // (Accessing parameters: pmCylinder->values[0])

    // CYLINDER: coordinate of a point located on the cylinder axis, a direction vector, and a radius
    pcl::ModelCoefficients::Ptr pmCylinder;
    pcl::PointIndices::Ptr cylinderInliers;

    // SPHERE: center point x,y,z, and a radius.
    pcl::ModelCoefficients::Ptr pmSphere;
    pcl::PointIndices::Ptr sphereInliers;

    // CUBOID
    cuboid pmCube_, pmCubeBB;
    pcl::PointIndices::Ptr cubeInliers;

    pcl::PointIndices::Ptr inliers;


    // GRASP FEATURE COMPUTING FUNCTIONS
    std::vector<double> computePosRelativeToAxes(const V3f &pos ) const;
    std::vector<double> computeFreeVolume(const PC::Ptr &graspedCloud) const;
    std::vector<double> computeGraspAngleWithUp(const V3f &approachVector) const;
    std::vector<double> computeElongatedness() const;
    std::vector<double> computeObjectVolume() const;
    std::vector<double> computeObjectDimensions() const;

    void findApproachVector(const PC::Ptr &cloudObj, const PC::Ptr &cloudGrasp);

    boost::shared_ptr<std::vector<pcl::Vertices> > vv;

};

#endif // MAINAXES_H
