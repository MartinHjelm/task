#include "saccuboid.h"

// std
#include <ctime>

// Boost
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
//#include <pcl/segmentation/sac_segmentation.h>

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

// Mine
#include <myhelperfuns.h>
#include <eigenhelperfuns.h>
#include <pclhelperfuns.h>

SacCuboid::SacCuboid() :
    maxDegWiggle(DEG2RAD(5)),
    distanceThreshold(0.0),
    normalDistanceWeight(0.0),
    maxIter(100)
{}


/*** SETTERS ***/
void
SacCuboid::setInputCloud(const PC::Ptr &cPtr) { cloud = cPtr; }

void
SacCuboid::setInputNormals(const PCN::Ptr &cnPtr) { cloudNormals = cnPtr; }

void
SacCuboid::setDistanceThreshold(double d) { if(d > 0) distanceThreshold = d; }

void
SacCuboid::setNormalDistanceWeight(double dw)
{
    if( dw >= 0.0 && dw <= 1.0 ) normalDistanceWeight = dw;
}

void
SacCuboid::setMaxIterations(uint n) { if(n > 0) maxIter = n; }




/*** MODEL ***/

int
SacCuboid::computeModel(Eigen::VectorXf &modelCoefficients)
{
    double angle = 0.0;
    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> dis(0, cloud->size());

    V4f p1,p2;
    for(int i_iter = 0; i_iter!=100; i_iter++)
    {

        // PLANE 1
        // 1. Pick random point, P1, in point cloud.
        p1Idx = dis(gen);
        p1 = cloudNormals->points[p1Idx].getNormalVector4fMap();
        p1.normalize();

        // 2. Find a random point, P2, orthogonal to P1, in the point cloud.
        for ( int j_iter = 0; j_iter != 100; j_iter++ )
        {
            p2Idx = dis(gen);
            p2 = cloudNormals->points[p2Idx].getNormalVector4fMap();
            p2.normalize();
            // Check if point normal is at a 90deg angle
            angle = std::fabs( M_PI_2 - acos( p1.dot(p2) ) );
            if(angle<maxDegWiggle) break;
        }
        if(angle>maxDegWiggle) continue; //Redo
        break;
    }

    // All sample points found. Compute model.
    modelCoefficients[0] = p1(0);
    modelCoefficients[1] = p1(1);
    modelCoefficients[2] = p1(2);

    modelCoefficients[4] = p2(0);
    modelCoefficients[5] = p2(1);
    modelCoefficients[6] = p2(2);

    V4f p1Pl1 = cloud->points[p1Idx].getVector4fMap();
    V4f p1Pl2 = cloud->points[p2Idx].getVector4fMap();

    /* Find the d1, d2 parameters for the plane equations. Use the
   * point's normals, n, as normals for the plane and the points as
   * points in the plane. Then we get: d = -npx*px - npy*py - npz*pz.
   */
    modelCoefficients(3) = -1.0 * (p1.dot(p1Pl1));
    modelCoefficients(7) = -1.0 * (p2.dot(p1Pl2));

    /** COMPUTE CUBOID PARAMS **/
//    pl1Params = modelParams.head<4>();
//    pl2Params = modelParams.tail<4>();

    /* After finding the two planes that best approximates a cuboid we
   * need to compute the plane's boundaries.
   */
//    if(!computeEnclosingRectangle(modelCoefficients.head<4>(),r1))
//        return 0;
//    if(!computeEnclosingRectangle(modelCoefficients.tail<4>(),r2))
//        return 0;

//    computeCuboidParams(modelCoefficients);
    return 1;
}

void
SacCuboid::selectWithinDistance(const Eigen::VectorXf &model_coefficients, pcl::PointIndices::Ptr &inliers)
{

    // Convert model coeffs to plane params
    V4f pl1 = model_coefficients.head<4>();
    V4f pl2 = model_coefficients.tail<4>();

    /* Compute the distance for each point to each of the planes. */
    for(uint i_point = 0; i_point!=cloud->size(); ++i_point)
    {
        std::vector<double> d_euclid(2,0.0), d_normal(2,0.0);
        // Compute min euclidian distance and angular distance to plane normals
        // A point's distance to the plane is the projection of the vector from origo
        // to the point onto the plane's normal plus the distance from the plane to origo.

        // Euclidian Distance
        d_euclid[0] = pointToPlaneDistance(cloud->points[i_point],pl1);
        d_euclid[1] = pointToPlaneDistance(cloud->points[i_point],pl2);

        // Normal distance
        d_normal[0] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl1);
        d_normal[1] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl2);

        // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
        double weight = normalDistanceWeight * (1.0 - cloudNormals->points[i_point].curvature);

        double dw1 = std::fabs(weight * d_normal[0] + (1 - weight) * d_euclid[0]);
        double dw2 = std::fabs(weight * d_normal[1] + (1 - weight) * d_euclid[1]);

        // Add point to inlier score
        double d = dw1 < dw2 ? dw1 : dw2;
        if(d < distanceThreshold){ inliers->indices.push_back(i_point); }

    }
}

void
SacCuboid::selectWithinDistance2(pcl::PointIndices::Ptr &inliers)
{

    /* Compute the distance for each point to each of the planes. */
    for(uint iPt = 0; iPt!=cloud->size(); ++iPt)
    {
        if( c_.isPtInlier(cloud->points[iPt].getVector3fMap(),distanceThreshold) )
            inliers->indices.push_back(iPt);
    }
}


void
SacCuboid::selectWithinDistance3(const cuboid &c, pcl::PointIndices::Ptr &inliers)
{
    std::cout << inliers->indices.size() << std::endl;
    for(uint i_point = 0; i_point!=cloud->size(); ++i_point)
    {
        double d_euclid = c.shortestSideDist(cloud->at(i_point).getVector3fMap());
        double d_angular = c.shortestAngularDist(cloud->at(i_point).getVector3fMap(),cloudNormals->at(i_point).getNormalVector3fMap());

        // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
//        double weight = normalDistanceWeight * (1.0 - cloudNormals->points[i_point].curvature);
        double d = std::fabs(normalDistanceWeight * d_angular + (1.0 - normalDistanceWeight) * d_euclid);
//        if(i_point>10586 && d < distanceThreshold)
//            printf("%i Angle: %f, Distance: %f, Distance: %f \n",i_point,RAD2DEG(d_angular),d_euclid,d);

        // Add point to inlier score
        if(d < distanceThreshold){
            inliers->indices.push_back(i_point); }
        }

    std::cout << inliers->indices.size() << std::endl;

}


int
SacCuboid::countWithinDistance3(const cuboid &c)
{
    int Ninliers = 0;
    for(uint i_point = 0; i_point!=cloud->size(); ++i_point)
    {
        double d_euclid = c.shortestSideDist(cloud->at(i_point).getVector3fMap());
        double d_angular = c.shortestAngularDist(cloud->at(i_point).getVector3fMap(),cloudNormals->at(i_point).getNormalVector3fMap());

        // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
//        double weight = normalDistanceWeight * (1.0 - cloudNormals->points[i_point].curvature);
        double d = std::fabs(normalDistanceWeight * d_angular + (1 - normalDistanceWeight) * d_euclid);

        // Add point to inlier score
        if(d < distanceThreshold){ Ninliers++; }
    }
    return Ninliers;
}


void
SacCuboid::selectWithinDistancePlane(const Eigen::VectorXf &modelCoefficients, std::vector<int> &inliers)
{
    /* Compute the distance for each point to each of the planes. */
    for(uint i_point = 0; i_point!=cloud->size(); ++i_point)
    {
        double d_euclid=0.0, d_normal=0.0;
        // Compute min euclidian distance and angular distance to plane normals
        // A point's distance to the plane is the projection of the vector from origo
        // to the point onto the plane's normal plus the distance from the plane to origo.

        // Euclidian Distance
        d_euclid = pointToPlaneDistance(cloud->points[i_point],modelCoefficients);

        // Normal distance
        d_normal = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], modelCoefficients);

        // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
        double weight = normalDistanceWeight * (1.0 - cloudNormals->points[i_point].curvature);

        double dw1 = std::fabs(weight * d_normal + (1 - weight) * d_euclid);
        double dw2 = std::fabs(weight * d_normal + (1 - weight) * d_euclid);

        // Add point to inlier score
        double d = dw1 < dw2 ? dw1 : dw2;
        if(d < distanceThreshold){ inliers.push_back(i_point); }
    }
}


int
SacCuboid::countWithinDistance(const Eigen::VectorXf &modelCoefficients)
{
    //  // Needs a valid set of model coefficients
    //  if (model_coefficients.size () != 8)
    //  {
    //    PCL_ERROR ("[pcl::SampleConsensusModelPlane::getDistancesToModel] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
    //    return 0;
    //  }

    // Convert model coeffs to plane params
    V4f pl1 = modelCoefficients.head<4>();
    V4f pl2 = modelCoefficients.tail<4>();

    /* Compute the distance for each point to each of the planes. */
    int counter = 0;
    std::vector<double> d_euclid(2,0.0), d_normal(2,0.0);
    for(uint i_point = 0; i_point!=cloud->size(); ++i_point)
    {
        // Compute min euclidian distance and angular distance to plane normals
        // A point's distance to the plane is the projection of the vector from origo
        // to the point onto the plane's normal plus the distance from the plane to origo.

        // Euclidian Distance
        d_euclid[0] = pointToPlaneDistance(cloud->points[i_point],pl1);
        d_euclid[1] = pointToPlaneDistance(cloud->points[i_point],pl2);

        // Normal distance
        d_normal[0] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl1);
        d_normal[1] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl2);

        // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
        double weight = normalDistanceWeight * (1.0 - cloudNormals->points[i_point].curvature);

        double dw1 = std::fabs(weight * d_normal[0] + (1 - weight) * d_euclid[0]);
        double dw2 = std::fabs(weight * d_normal[1] + (1 - weight) * d_euclid[1]);

        // Add point to inlier score
        double d = dw1 < dw2 ? dw1 : dw2;
        if(d < distanceThreshold){ counter++; }
    }

    //  std::cout << "OK! Count Within Distance" << std::endl;

    return counter;
}

int
SacCuboid::countWithinDistance2(const V4f &pl1,const V4f &pl2,const V4f &pl3)
{
    /* Compute the distance for each point to each of the planes. */
    int counter = 0;
    std::vector<double> d_euclid(3,0.0), d_normal(3,0.0);
    for(uint i_point = 0; i_point!=cloud->size(); ++i_point)
    {
        // Compute min euclidian distance and angular distance to plane normals
        // A point's distance to the plane is the projection of the vector from origo
        // to the point onto the plane's normal plus the distance from the plane to origo.

        // Euclidian Distance
        d_euclid[0] = pointToPlaneDistance(cloud->points[i_point],pl1);
        d_euclid[1] = pointToPlaneDistance(cloud->points[i_point],pl2);
        d_euclid[2] = pointToPlaneDistance(cloud->points[i_point],pl3);

        // Normal distance
        d_normal[0] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl1);
        d_normal[1] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl2);
        d_normal[2] = pointNormalToPlaneNormalDistance(cloudNormals->points[i_point], pl3);

        // Weight with the point curvature. On flat surfaces, curvature -> 0, which means the normal will have a higher influence
        double weight = normalDistanceWeight * (1.0 - cloudNormals->points[i_point].curvature);

        double dw1 = std::fabs(weight * d_normal[0] + (1 - weight) * d_euclid[0]);
        double dw2 = std::fabs(weight * d_normal[1] + (1 - weight) * d_euclid[1]);
        double dw3 = std::fabs(weight * d_normal[2] + (1 - weight) * d_euclid[2]);

        // Add point to inlier score
        double d = dw1 < dw2 ? dw1 : dw2;
        d = dw3 < d ? dw3 : d;
        if(d < distanceThreshold){ counter++; }
    }

    //  std::cout << "OK! Count Within Distance" << std::endl;

    return counter;
}


bool
SacCuboid::isModelOK(const Eigen::VectorXf &modelCoefficients)
{
    // Convert model coeffs to plane params
    V3f n1 = modelCoefficients.head(3);
    V3f n2 = modelCoefficients.segment(4,3);

    if(std::abs( M_PI_2 - acos( n1.dot(n2) ) ) >= maxDegWiggle) { return false; }
    else return true;
}

void
SacCuboid::sacFitCuboid2PointCloud( Eigen::VectorXf &modelCoefficients )
{
    modelCoefficients = Eigen::VectorXf::Zero(8);
    Eigen::VectorXf params = Eigen::VectorXf::Zero(8);
    int bestScore = 0;

    // RANSAC
    for ( uint i_iter = 0; i_iter < maxIter; i_iter++ )
    {
        int score = 0;
        if( computeModel(params) )
            score = countWithinDistance(params);
            //score = countWithinDistance3(c_);

//        printf("Score: %i\n", score);

        // If best fit sofar keep points and normals
        if(score > bestScore)
        {
            // Save values
            modelCoefficients = params;
            p1IdxBest = p1Idx;
            p2IdxBest = p2Idx;
            bestScore = score;
        }
    }

    /* After finding the two planes that best approximates a cuboid we
     * need to compute the plane's boundaries.
     */
    if(!computeEnclosingRectangle(modelCoefficients.head<4>(),r1))
        return;
    if(!computeEnclosingRectangle(modelCoefficients.tail<4>(),r2))
        return;

    // Compute cuboid params from plane model
    computeCuboidParams(modelCoefficients);
}


// Computes the enclosing rectangle of a plane.
int
SacCuboid::computeEnclosingRectangle(const Eigen::Vector4f &planeParams, std::vector<Eigen::Vector3f> &rPoly)
{

//    printf("Starting rectangle computations: \n"); fflush(stdout);

    rPoly.empty();

    // 1. Get inliers for this side of the cube.
    std::vector<int> ptsCubeSide1;
    selectWithinDistancePlane(planeParams,ptsCubeSide1);
    if(ptsCubeSide1.size()==0)
        return 0;


//    printf("1: \n"); fflush(stdout);

    /* 2. Create a coordinate system which has a xy-axis along the plane. The plane normal is one axis,
   * the second is a vector in the plane, and the cross product of the two is the third axis.
   * Compute the transform matrix and inverse transform matrix for that coordinate system.
   */
    // Axes matrices for the planes
    Eigen::Matrix3f transformMat, transformMatInverse;

    // Normal as the third axis
    for(int idx=0;idx<3;idx++) transformMat(2,idx) = planeParams[idx];

    // Vector in the plane as the first axis, i.e., P2-P1.
    transformMat(0,0) = 0.0;
    transformMat(0,1) = -planeParams[1]/planeParams[3];
    transformMat(0,2) = planeParams[2]/planeParams[3];

    // Take cross product of the two as the second axis.
    transformMat.row(1) = transformMat.row(2).cross(transformMat.row(0));

    transformMat.row(0) = transformMat.row(0).normalized();
    transformMat.row(1) = transformMat.row(1).normalized();
    transformMat.row(2) = transformMat.row(2).normalized();
    transformMatInverse = transformMat.inverse();

//    printf("2: \n"); fflush(stdout);

    /* 3. Transform all points to the new plane centered coordinate system.
   * And pick out the vector coordinates in the plane. These will be fed to
   * the opencv findminrect function.
   */
    Eigen::Vector3f point;
    //  normal = planeParams.head<3>();
    //  origo = cloud->points[p1IdxBest].getVector3fMap();
    cv::Point2f pt; // Opencv point representation
    std::vector<cv::Point2f> points;


    V3f origo2 = cloud->points[ptsCubeSide1[0]].getVector3fMap();

//    printf("3: \n"); fflush(stdout);

//    std::cout << "Cloud size: " << cloud->size() << std::endl;

    for( std::vector<int>::iterator ptIdx = ptsCubeSide1.begin(); ptIdx != ptsCubeSide1.end(); ++ptIdx)
    {
        // Get point in the point cloud
        point = cloud->points[*ptIdx].getVector3fMap();
        point = transformMat * ( point - origo2);
        pt.x = point(0); pt.y = point(1);
        points.push_back(pt);
    }

//    printf("Points in rect: %lu\n", points.size()); fflush(stdout);


    // 4. Find the min area rectangle in the 2-dim space
    cv::RotatedRect rrect = cv::minAreaRect(cv::Mat(points) );
    cv::Point2f rrPts[4];
    rrect.points(rrPts); // Convert to a rotade rectangle.



    // 5. Transform the 2D rectangle points back to 3D points in the
    // point cloud coordinate system.
    // Add the points to the rectangle vector
    V3f rpt(0,0,0);
    for( int ptIdx = 0; ptIdx < 4; ptIdx++ )
    {
        rpt(0) = rrPts[ptIdx].x; rpt(1) = rrPts[ptIdx].y; rpt(2) = 0.0;
        rpt = transformMatInverse * rpt + origo2;
        rPoly.push_back(rpt);
    }

//    printf("Points in poly1: %lu\n", rPoly.size()); fflush(stdout);

    // Add the center of the rectangle
    rpt(0) = rrect.center.x; rpt(1) = rrect.center.y; rpt(2) = 0.0;
    rpt = transformMatInverse * rpt + origo2;
    rPoly.push_back(rpt);

//    printf("Points in poly2: %lu\n", rPoly.size()); fflush(stdout);

    return 1;

}



void
SacCuboid::computeCuboidParams(const Eigen::VectorXf &model_coefficients)
{

    // Set cuboid main axes
    Eigen::Vector3f nrmlVec1,nrmlVec2,nrmlVec3;
    nrmlVec1 = model_coefficients.head<3>();
    c_.axisMat.col(0) = nrmlVec1;
    nrmlVec2 = model_coefficients.tail<3>();
    c_.axisMat.col(1) = nrmlVec2;
    nrmlVec3 = nrmlVec1.cross(nrmlVec2);
    nrmlVec3.normalize();
    c_.axisMat.col(2) = nrmlVec3;

    // Find the min/max of the cube sides by projecting the points down on each of the cube axes
    // Cube width, height, and depth
    std::vector<double> prjVals(3,0.0);
    std::vector<double> prjMins(3,1000000.0);
    std::vector<double> prjMaxs(3,-1000000.0);
    for(PC::iterator iter = cloud->begin(); iter!=cloud->end(); ++iter)
    {
        Eigen::Vector3f p = iter->getVector3fMap();

        prjVals[0] = ( nrmlVec1.dot( p ) );
        prjVals[1] = ( nrmlVec2.dot( p ) );
        prjVals[2] = ( nrmlVec3.dot( p ) );

        for( int idx = 0; idx < 3; idx++ )
        {
            if(prjVals[idx]>prjMaxs[idx]) prjMaxs[idx] = prjVals[idx];
            if(prjVals[idx]<prjMins[idx]) prjMins[idx] = prjVals[idx];
        }
    }

    c_.width = std::abs(prjMaxs[0]-prjMins[0]);
    c_.height = std::abs(prjMaxs[1]-prjMins[1]);
    c_.depth = std::abs(prjMaxs[2]-prjMins[2]);


    // Order by length
    if(c_.height < c_.depth )
    {
        double tmp = c_.height;
        Eigen::Vector3f tmpVec = c_.axisMat.col(1);
        c_.height = c_.depth;
        c_.axisMat.col(1) = c_.axisMat.col(2);
        c_.depth = tmp;
        c_.axisMat.col(2) = tmpVec;
    }

    if(c_.width < c_.height )
    {
        double tmp = c_.width;
        Eigen::Vector3f tmpVec = c_.axisMat.col(0);

        c_.width = c_.height;
        c_.axisMat.col(0) = c_.axisMat.col(1);
        c_.height = tmp;
        c_.axisMat.col(1) = tmpVec;
    }

    if(c_.height < c_.depth )
    {
        double tmp = c_.height;
        Eigen::Vector3f tmpVec = c_.axisMat.col(1);
        c_.height = c_.depth;
        c_.axisMat.col(1) = c_.axisMat.col(2);
        c_.depth = tmp;
        c_.axisMat.col(2) = tmpVec;
    }
    //*/


    // This REALLY needs fixing......
    /***********************************/

    c_.transVec = r1[4];
    // Project all points in the point cloud down on this and then take max
    nrmlVec1 = model_coefficients.head<3>();
    double prjVal=0.0; double prjMax=0.0;
    for( uint i_point = 0; i_point < cloud->size(); i_point++ )
    {
        prjVal = std::abs( nrmlVec1.dot( cloud->points[i_point].getVector3fMap()-c_.transVec ) );
        if(prjVal > prjMax) prjMax = std::abs(prjVal);
    }
    c_.transVec -= (0.5*prjMax)*nrmlVec1;
    /***********************************/

    // Create quartention vector
    Eigen::Quaternionf cc(c_.axisMat);
    c_.quartVec = cc;

//    c_.width = cc._transformVector(c_.axisMat.row(0)*c_.width).norm();
//    c_.height = cc._transformVector(c_.axisMat.row(1)*c_.height).norm();
//    c_.depth = cc._transformVector(c_.axisMat.row(2)*c_.depth).norm();

//    Eigen::Matrix3f axisMatrix = Eigen::Matrix3f::Identity();
//    axisMatrix.row(0) = cc._transformVector(axisMatrix.row(0));
//    axisMatrix.row(1) = cc._transformVector(axisMatrix.row(1));
//    axisMatrix.row(2) = cc._transformVector(axisMatrix.row(2));
//    c_.axisMat.col(0) = axisMatrix.row(0).transpose();
//    c_.axisMat.col(1) = axisMatrix.row(1).transpose();
//    c_.axisMat.col(2) = axisMatrix.row(2).transpose();
    c_.setAxisRotMat();
   // c_.setXYZminmax();
}




/* Point to plane distance, that is, d_euclid = abs(a*x+b*x+c*x+d) */
double
SacCuboid::pointToPlaneDistance (const PointT &p, const V4f &plParams)
{
    V4f pVec = p.getVector4fMap(); pVec(3)=1;
    return std::fabs(plParams.dot(pVec));;
}


/* Angular distance between point normal and plane normal. */
double
SacCuboid::pointNormalToPlaneNormalDistance (const PointN &p, const V4f &plane_coefficients)
{
    V4f n = plane_coefficients;
    n(3) = 0;
    double d_normal = std::fabs( pcl::getAngle3D(n, p.getNormalVector4fMap()) );
    return (std::min) (d_normal, M_PI - d_normal);
}
