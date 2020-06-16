#include "mainaxes.h"

// STD
#include <iterator>
#include <algorithm>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

// Mine
#include <saccuboid.h>
#include <myhelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>

MainAxes::MainAxes() :
    printMsg(false),
    normalDistanceWeight(0.0),
    distanceThreshold(0.0),
    radiusLimit(0.0),
    axesLengths(0,0,0),
    midPoint(0,0,0),
    pmCylinder(new pcl::ModelCoefficients),
    cylinderInliers (new pcl::PointIndices),
    pmSphere(new pcl::ModelCoefficients),
    sphereInliers (new pcl::PointIndices),
    cubeInliers (new pcl::PointIndices),
    inliers (new pcl::PointIndices),
    vv(new std::vector<pcl::Vertices>)
{}

void
MainAxes::setInputSource(const PC::Ptr &cloud, const PCN::Ptr &cloudNormals, V3f tabelNormal)
{
    cloud_ = cloud; // Make copy of clouds
    cloudNormals_ = cloudNormals;
    tableNrml = tabelNormal;
}




/************ FIND AXIS + SHAPE PRIMITIVE SAC FITTING MODELS ************/


void
MainAxes::fitObject2Primitives()
{

    fitScores.clear();

    // Get bounding box to limit size/radius of ransac cyldinders and spheres
    computePointCloudBoundingBox();
    setMaxRadius();

    // 1. Fit cylinder
    fitScores.push_back( fitCylinder2PointCloud() );

    // 2. Fit sphere.
    fitScores.push_back( fitSphere2PointCloud() );

    // 3. Fit to cuboid
    fitScores.push_back( fitCuboid2PointCloud() );

    printMsg=true;
    MyHelperFuns::printString("RANSAC fitting algorithm scores: ", printMsg);
    MyHelperFuns::printString("Cylinder: "+MyHelperFuns::toString(fitScores[0]), printMsg);
    MyHelperFuns::printString("Sphere: "+MyHelperFuns::toString(fitScores[1]), printMsg);
    MyHelperFuns::printString("Cuboid: "+MyHelperFuns::toString(fitScores[2]), printMsg);
    printMsg=false;

    // Object primitive
    objectPrimitive = std::distance(fitScores.begin(), std::max_element(fitScores.begin(), fitScores.end()) );
    double bestScore = fitScores[objectPrimitive];
    // Fit scores
    for(std::vector<double>::iterator iValPtr=fitScores.begin();iValPtr!=fitScores.end();iValPtr++)
        *iValPtr = *iValPtr/cloud_->size();

    if(bestScore<250 || fitScores[objectPrimitive]<0.01)
    {
        printf("FAILED to find any good primitives defaulting to cuboid!\n");
        computeFeatureParamsFromCuboid();
        inliers = cubeInliers;
        //    throw std::runtime_error( "No matching primitive found!" );
    }
    else
    {
        switch (objectPrimitive)
        {
        case 0:
            computeFeatureParamsFromCylinder();
            inliers = cylinderInliers;
            break;
        case 1:
            computeFeatureParamsFromSphere();
            inliers = sphereInliers;
            break;
        case 2:
            computeFeatureParamsFromCuboid();
            inliers = cubeInliers;
            break;
        default:
            printf("Defaulted on the type of object primitives, defaulting to cylinder..\n");
            computeFeatureParamsFromCylinder();
            inliers = cubeInliers;
            break;
        }
    }

}


/************ COMPUTE AXIS FROM PRIMITIVES FUNS ************/

void
MainAxes::computeFeatureParamsFromCylinder()
{
    /* Cylinder model is based on a line and radius:
   * a point on the line,
   * a direction vector of the line,
   * a radius r
   */

    // Project all points onto the direction vector of the line
    // that defines the cylinder, and find the max and the min.
    V3f vCylDir, pLine, inlierPoint;

    vCylDir << pmCylinder->values[3], pmCylinder->values[4], pmCylinder->values[5];
    vCylDir.normalize();

    pLine << pmCylinder->values[0],pmCylinder->values[1],pmCylinder->values[2];


    // Set axis lengths - Eigen::Vector3f axesLengths;
    std::vector<double> projVals;
    std::vector<int>::const_iterator iter = cylinderInliers->indices.begin();
    for(; iter != cylinderInliers->indices.end(); ++iter)
    {
        inlierPoint = cloud_->points[*iter].getVector3fMap();
        projVals.push_back(vCylDir.dot(inlierPoint-pLine));
    }

    float min = MyHelperFuns::minValVector(projVals);
    float max = MyHelperFuns::maxValVector(projVals);
    axesLengths(0) = (max-min)/2;
    axesLengths(1) = pmCylinder->values[6];
    axesLengths(2) = pmCylinder->values[6];
    midPoint = pLine + (min+axesLengths(0)) * vCylDir;

    // Set object axes vectors - width, height, depth
    V3f a2((vCylDir(1)/vCylDir(0)),-1,0);
    a2.normalize();
    V3f a3;
    a3 = vCylDir.cross(a2);
    a3.normalize();
    axesVectors.col(0) = vCylDir;
    axesVectors.col(1) = a2;
    axesVectors.col(2) = a3;

    // Rotate axis vector 180deg if it is pointing downwards
    for(int i=0; i!=3; i++)
        if( axesVectors.col(i).dot(tableNrml) < 0 )
            axesVectors.col(i) = -axesVectors.col(i);

    // Note to self. Direction of cylinder depends on vCylDir sometimes it is in the
    // opposite direction of the table the object is standing on.
}

void
MainAxes::computeFeatureParamsFromSphere()
{
    // Set axis lengths - V3f axesLengths;
    for( int idx = 0; idx < 3; idx++ )
    {
        axesLengths(idx) = pmSphere->values[3];
    }

    // Set object mid-point - Eigen::Vector3f midPoint
    for(int idx = 0; idx < 3; idx++)
    {
        midPoint(idx) = pmSphere->values[idx];
    }


    V3f n1 = tableNrml;
    n1.normalize();
    V3f n2(-1,n1(0)/n1(1),0);
    n2.normalize();
    V3f n3 = n1.cross(n2);
    n3.normalize();
    EigenHelperFuns::gmOrth(n1,n2,n3);

    axesVectors.col(0) = n1;
    axesVectors.col(1) = n2;
    axesVectors.col(2) = n3;

    // Rotate axis vector 180deg if it is pointing downwards
    for(int i=0; i!=3; i++)
        if( axesVectors.col(i).dot(tableNrml) < 0 )
            axesVectors.col(i) = -axesVectors.col(i);
}

void
MainAxes::computeFeatureParamsFromCuboid()
{
    midPoint = pmCube_.transVec;
    axesVectors = pmCube_.axisMat;
    axesLengths(0) = 0.5f*pmCube_.width;
    axesLengths(1) = 0.5f*pmCube_.height;
    axesLengths(2) = 0.5f*pmCube_.depth;

    // Rotate main axis vector 180deg if it is pointing downwards
    if( axesVectors.col(0).dot(tableNrml) < 0 )
        axesVectors.col(0) = -axesVectors.col(0);
}



/************ PRIMTIVE FITTING FUNCTIONS ************/

double
MainAxes::fitCylinder2PointCloud()
{
    /* Cylinder model is based on:
   * a vector on a line,
   * a direction vector of the line,
   * a radius r
   */

    // Clear variables
    cylinderInliers->indices.clear();

    pcl::SACSegmentationFromNormals<PointT, PointN> seg;

    // Create the segmentation object for cylinder segmentation and set all the parameters
    seg.setOptimizeCoefficients ( true);
    seg.setModelType (pcl::SACMODEL_CYLINDER);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (500000);
    seg.setDistanceThreshold (0.03+DEG2RAD(5));
//    seg.setNormalDistanceWeight (0.5);
    seg.setRadiusLimits (0.001, cylRadiusMax);
    seg.setInputCloud (cloud_);
    seg.setInputNormals (cloudNormals_);
    // Obtain the cylinder inliers and coefficients
    seg.segment(*cylinderInliers, *pmCylinder);
    // Return fit score as the number of points that are inliers to the model.
    return cylinderInliers->indices.size();
}

double
MainAxes::fitSphere2PointCloud() {
    /* Sphere model is based on:
   * a central point
   * a radius r
   */
    pcl::SACSegmentationFromNormals<PointT, PointN> seg;

    // Clear variables
    sphereInliers->indices.clear();

    // Create the segmentation object for cylinder segmentation and set all the parameters
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_SPHERE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (50000);
    seg.setDistanceThreshold (0.01+DEG2RAD(1));
    seg.setNormalDistanceWeight (0.6);
    seg.setRadiusLimits (0.001, sphRadiusMax);
//    std::cout << sphRadiusMax << std::endl;
    seg.setInputCloud (cloud_);
    seg.setInputNormals (cloudNormals_);
    // Obtain the cylinder inliers and coefficients
    seg.segment (*sphereInliers, *pmSphere);
//    printf("Sphere inliers: %lu", sphereInliers->indices.size());
    // Return fit score as the number of points that are inliers to the model.
    return sphereInliers->indices.size();
}

double
MainAxes::fitCuboid2PointCloud()
{
    // Clear variables
    cubeInliers->indices.clear();

    // Sac-Cuboid fitting
    SacCuboid SA;
//    Eigen::VectorXf modelParams;
    SA.setInputCloud(cloud_);
    SA.setInputNormals(cloudNormals_);
    SA.setMaxIterations(1);
    SA.setDistanceThreshold(0.03+DEG2RAD(5)); // DEG2RAD(5)
    SA.setNormalDistanceWeight(.5);
//    SA.sacFitCuboid2PointCloud(modelParams);

    // Do trials between pca approach and ransac approach
    PCLHelperFuns::fitCuboid2PointCloudPlane(cloud_, tableNrml, pmCube_);
//    pmCube_.print();
    int Ninliers1 = SA.countWithinDistance3(pmCube_);
//    cuboid tmpCube = SA.getModelParameters();
//    int Ninliers2 = SA.countWithinDistance3(tmpCube);
//    int Ninliers2 = SA.countWithinDistancemodelParams);
//    std::cout << "Number of box inliers(1): " << Ninliers1 << std::endl;
//    std::cout << "Number of box inliers(2): " << Ninliers2 << std::endl;

//    if(true)//cloud_->size() < 1000 || Ninliers1 > Ninliers2 ) // || (tmpCube.width > cylRadiusMax || tmpCube.height > cylRadiusMax || tmpCube.depth > cylRadiusMax) )
//    {
        SA.selectWithinDistance3(pmCube_, cubeInliers);
        return double(Ninliers1);
//    }
//    else
//    {
//        tmpCube.cpyCuboid(pmCube_);
//        SA.selectWithinDistance3(pmCube_, cubeInliers);
//        return (double)Ninliers1;
//    }

//    return Ninliers2;
}

void
MainAxes::computePointCloudBoundingBox()
{
    // compute principal direction
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_, centroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*cloud_, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

    // move the points to the that reference frame
    Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
    p2w.block<3,3>(0,0) = eigDx.transpose();
    p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
    PC cPoints;
    pcl::transformPointCloud(*cloud_, cPoints, p2w);

    PointT min_pt, max_pt;
    pcl::getMinMax3D(cPoints, min_pt, max_pt);
    V3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());

    // final transform
    Eigen::Quaternionf qfinal(eigDx);
    qfinal.normalize();
    V3f tfinal = eigDx*mean_diag + centroid.head<3>();

    pmCubeBB.axisMat = eigDx;
    pmCubeBB.transVec = tfinal;
    pmCubeBB.quartVec = qfinal;
    pmCubeBB.width = max_pt.x - min_pt.x;
    pmCubeBB.height = max_pt.y - min_pt.y;
    pmCubeBB.depth = max_pt.z - min_pt.z;

    pmCubeBB.x_min = -0.5f*pmCubeBB.width; pmCubeBB.x_max = 0.5f*pmCubeBB.width;
    pmCubeBB.y_min = -0.5f*pmCubeBB.height; pmCubeBB.y_max = 0.5f*pmCubeBB.height;
    pmCubeBB.z_min = -0.5f*pmCubeBB.depth; pmCubeBB.z_max = 0.5f*pmCubeBB.depth;
}

void
MainAxes::setMaxRadius()
{
    std::vector<double> dimVals;
    dimVals.push_back(pmCubeBB.width);
    dimVals.push_back(pmCubeBB.height);
    dimVals.push_back(pmCubeBB.depth);
    double max = MyHelperFuns::maxValVector(dimVals) / 2;
    cylRadiusMax = max;
    sphRadiusMax = max;
}





/************ FEATURE FUNCTIONS ************/

/* Computes the relative position of a gripper wrt to the main axes. With
 * respect to a coordinate system in the middle of the object.
*/
std::vector<double>
MainAxes::computePosRelativeToAxes(const V3f &pos) const
{
    std::vector<double> relPos(3,0.0);
    V3f ptProj = (Eigen::MatrixXf::Identity(3,3).transpose()*axesVectors) * (pos-midPoint);
    for( int idx = 0; idx != 3; idx++ )
    {
        relPos[idx] = 100 * (ptProj(idx)/axesLengths(idx));
    }
    return relPos;
}



/* Computes the ratio of the volume of the part of the object that is contained
 * in the hand and the volume of the object bounding box. The volume is computed
 * by dividing the grasping cuboid into small cubes and computing the volume of
 * the cubes laying inside the object.
*/
std::vector<double>
MainAxes::computeFreeVolume( const PC::Ptr &graspedCloud ) const
{

    // Compute volume of object
    PC::Ptr cloudObject(new PC);
    pcl::ConvexHull<PointT> chull;
    chull.setInputCloud(cloud_);
    chull.setComputeAreaVolume(true);
    chull.setDimension(3);
    chull.reconstruct(*cloudObject);
//    printf("The volume of the object: %f\n",chull.getTotalVolume());
    double v1 = chull.getTotalVolume();

    PC::Ptr cloudGraspedPart(new PC);
    pcl::ConvexHull<PointT> chull2;
    chull2.setInputCloud(graspedCloud);
    chull2.setComputeAreaVolume(true);
    chull2.setDimension(3);
    chull2.reconstruct(*cloudGraspedPart,*vv);
//    printf("The volume of the grasped part: %f\n",chull.getTotalVolume());
    double v2 = chull2.getTotalVolume();

    std::vector<double> volumes(2,0.0);
    volumes[0] = v1;
    volumes[1] = v2/v1;
    return volumes;

//    PC::Ptr hullGrasp(new PC);
//    void getCornersOfCuboidAsPC( PC::Ptr &cloud )



//    V3f hVec=cube.axisMat.row(1);
//    V3f wVec=cube.axisMat.row(0);
//    V3f dVec=cube.axisMat.row(2);

//    double stepSize = 0.01;
//    int stepsH = cube.height/stepSize;
//    int stepsW = cube.width/stepSize;
//    int stepsD = cube.depth/stepSize;

//    V3f pBottom = cube.transVec - (cube.height/2) * hVec - (cube.width/2) * wVec - (cube.depth/2) * dVec;
//    int Nsmallcubes = 0;
//    std::vector<double> posRel;
//    for(int wStep=0; wStep<stepsW; wStep++)
//        for(int hStep=0; hStep<stepsH; hStep++)
//            for(int dStep=0; dStep<stepsD; dStep++)
//            {
//                posRel = computePosRelativeToAxes(pBottom + (wStep*stepSize)*wVec + (hStep*stepSize)*hVec + (dStep*stepSize)*dVec);
//                if((posRel[0]>0 && posRel[1]>0 && posRel[2]>0) && (posRel[0]<100 && posRel[1]<100 && posRel[2]<100) )
//                    Nsmallcubes++;
//            }

//    double volumeObj = pmCubeBB.width*pmCubeBB.height*pmCubeBB.depth;
//    return std::vector<double>(1,std::fabs( (volumeObj-Nsmallcubes*(std::pow(stepSize,3)))/volumeObj));
}


std::vector<double>
MainAxes::computeElongatedness() const
{
    std::vector<double> elonVals(2,0.0);
    std::vector<double> alens;
    EigenHelperFuns::eigenVec2StdVec(axesLengths,alens);
    std::sort( alens.begin(),alens.end() );
    elonVals[0] = alens[1]/alens[0];
    elonVals[1] = alens[2]/alens[0];
    return elonVals;
}


/** Computes the angle of the vector between grasp cuboid center and object,
 * and the up vector in the robots coordinate system(the table normal in this case).  **/
std::vector<double>
MainAxes::computeGraspAngleWithUp( const V3f &approachVector ) const
{
    return std::vector<double>(1,EigenHelperFuns::angleBetweenVectors(approachVector,tableNrml));
}


// Returns the volume of the object
std::vector<double>
MainAxes::computeObjectVolume() const
{
    PC::Ptr cloudObject(new PC);
    pcl::ConvexHull<PointT> chull;
    chull.setInputCloud(cloud_);
    chull.setComputeAreaVolume(true);
    chull.setDimension(3);
    chull.reconstruct(*cloudObject);
//    printf("The volume of the object: %f\n",chull.getTotalVolume());
    double v1 = chull.getTotalVolume();
    return std::vector<double>(1,v1);

//    return std::vector<double>(1,pmCubeBB.width*pmCubeBB.height*pmCubeBB.depth);
}

// Returns the volume of the object
std::vector<double>
MainAxes::computeObjectDimensions() const
{
    std::vector<double> axLengths;
    EigenHelperFuns::eigenVec2StdVec(axesLengths,axLengths);
    std::sort( axLengths.begin(), axLengths.end() );
    return axLengths;
}




void
MainAxes::findApproachVector(const PC::Ptr &cloudObj, const PC::Ptr &cloudGrasp )
{
    PC::Ptr handCloud(new PC);
    PCN::Ptr handCloudNormals(new PCN);

    // Remove object from gasping scene
    // pcl::PointIndices::Ptr ptIdxs(new pcl::PointIndices);
    // PCLHelperFuns::cloud2origInd(cloudObj, ptIdxs);
    // PCLHelperFuns::filterOutByOrigID(ptIdxs, cloudGrasp, handCloud);



    PCLHelperFuns::filterCloudFromCloud(cloudGrasp,cloudObj,handCloud);
    handCloud->is_dense = false;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*handCloud,*handCloud, indices);
    handCloud->is_dense = true;


    cuboid graspBB;
    PCLHelperFuns::computePointCloudBoundingBox(cloudObj,graspBB);
    pcl::PointIndices::Ptr objIndices(new pcl::PointIndices);
    PCLHelperFuns::selectPointsInCube( cloudGrasp, graspBB, objIndices );

    pcl::ExtractIndices<PointT> extract;
    extract.setNegative(true);
    extract.setInputCloud(cloudGrasp);
    extract.setIndices(objIndices);
    extract.filter(*handCloud);



    PCLHelperFuns::computeCloudNormals(handCloud,0.01,handCloudNormals);

    std::cout << "Points in PC after removing object: " << handCloud->size() << std::endl;





    // pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // pcl::ModelCoefficients::Ptr pm (new pcl::ModelCoefficients);
    // pcl::SACSegmentationFromNormals<PointT, PointN> seg;
    // Create the segmentation object for cylinder segmentation and set all the parameters
//    seg.setOptimizeCoefficients(true);
    // for(int trial = 0; trial < 5; trial++)
    // {
    //     // seg.setModelType (pcl::SACMODEL_CYLINDER);
    //     seg.setModelType (pcl::SACMODEL_LINE);
    //     seg.setMethodType (pcl::SAC_RANSAC);
    //     seg.setMaxIterations (500000);
    //     seg.setDistanceThreshold  (0.03+DEG2RAD(5));
    // //    seg.setNormalDistanceWeight (0.);
    //     seg.setRadiusLimits (0.001, 0.15);
    //     seg.setInputCloud (handCloud);
    //     seg.setInputNormals (handCloudNormals);
    //     // Obtain the cylinder inliers and coefficients
    //     seg.segment(*inliers, *pm);
    //     std::cout << "Points in SAC: " << inliers->indices.size() << std::endl;
    //     if(inliers->indices.size()>0) break;
    // }
//    pcl::SampleConsensusModelCylinder<PointT, PointN>::Ptr model(new pcl::SampleConsensusModelCylinder<PointT, PointN>(handCloud));
//    model->setInputNormals(handCloudNormals);
//    pcl::RandomSampleConsensus<PointT> sac (model, 0.01);
//    bool result = sac.computeModel();
//    std::vector<int> sample;
//    sac.getModel(sample);
//    std::vector<int> inliers2;
//    sac.getInliers (inliers2);
//    Eigen::VectorXf coeff;
//    sac.getModelCoefficients(coeff);


    // compute principal direction
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*handCloud, centroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*handCloud, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

    std::cout << "eigen values " << eigen_solver.eigenvalues() << std::endl;

    approachVector(0) = eigDx.col(2)(0);
    approachVector(1) = eigDx.col(2)(1);
    approachVector(2) = eigDx.col(2)(2);
    approachVector.normalize();

    if(tableNrml.dot(approachVector) > 0)
        approachVector = -1*approachVector;


    V3f pLine;
    // pLine << pmCylinder->values[0],pmCylinder->values[1],pmCylinder->values[2];
    pLine << approachVector(0),approachVector(1),approachVector(2);


    // Set axis lengths - Eigen::Vector3f axesLengths;
    std::vector<double> projVals;
    // std::vector<int>::const_iterator iter = inliers->indices.begin();
    for (PC::iterator ptIter = handCloud->begin(); ptIter != handCloud->end(); ++ptIter)
    {
    // for(; iter != inliers->indices.end(); ++iter)
    // {
        V3f inlierPoint = ptIter->getVector3fMap();
        projVals.push_back(approachVector.dot(inlierPoint-pLine));
    }

    float min = MyHelperFuns::minValVector(projVals);
    float max = MyHelperFuns::maxValVector(projVals);
    approachVectorLength = (max-min);
    approachVectorStart = pLine;

}
