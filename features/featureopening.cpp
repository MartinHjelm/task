#include "featureopening.h"

// std
#include <vector>
#include <string>
#include <fstream>
//#include <ctime>
//#include <random>
#include <stdexcept>
#include <cmath>
#include <math.h>

// Boost instead of C++11
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// PCL headers
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>

// opencv
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// My SAC 3D circle with normal
#include <sac3dcirclewithnormal.h>
#include <myhelperfuns.h>
#include <eigenhelperfuns.h>
#include <pclhelperfuns.h>

FeatureOpening::FeatureOpening() :
    pmCircle(new pcl::ModelCoefficients),
    circleInliers (new pcl::PointIndices),
    hasOpening(false)
{}

void
FeatureOpening::setInputSource(const PC::Ptr &cldPtr, const PCN::Ptr &cldNrmlPtr, const MainAxes &MA, const cv::Mat& imgWindow)
{
    cloud = cldPtr;
    cloudNormals = cldNrmlPtr;
    mainAxis_ = MA.tableNrml;
    mainAxis_.normalize();
    imgWindow.copyTo(sIgm);
    objcenter_ = MA.midPoint;

    if ( MA.axesLengths(1) > MA.axesLengths(2) )
        maxRadius_ = MA.axesLengths(1);
    else
        maxRadius_ = MA.axesLengths(2);
}



void
FeatureOpening::detectOpening()
{
    // Fit a 3D circle to the object
    try { circle3DFit(); }
    catch (const std::exception& e) { std::cout << "Exception: " << e.what(); return;}

    //detect2DEllipses();
//    if ( !isCircleGood() )
//    {
//        printf("Sample no good!\n");
//        return;
//    }

    // Assign to Eigen representation to write less code..
    cCenter = V4f(pmCircle->values[0],pmCircle->values[1],pmCircle->values[2],0);
    cNormal = V4f(pmCircle->values[4],pmCircle->values[5],pmCircle->values[6],0);
    cNormal.normalize();
    cNormal(3) = -1 * cNormal.dot(cCenter);
    radius = pmCircle->values[3];

    int N_pp = 0;
    V3f p;
    V3f center = cCenter.head<3>();
    V3f n1 = cNormal.head<3>();
    V3f n2(-1,n1(0)/n1(1),0);
    n2.normalize();
    V3f n3 = n1.cross(n2);
    n3.normalize();
    EigenHelperFuns::gmOrth(n1,n2,n3);


    std::vector<float> degreeHist(360,0);
//    std::vector<float> pointDistHist(std::round(radius/0.01),0);
//    std::vector<float> pointDistHist(360,0);

    for(PC::iterator iter = cloud->points.begin(); iter != cloud->points.end(); ++iter)
    {
        // Move point to circle center and remove normal component
        p = iter->getVector3fMap() - center;
        double np = n1.dot(p);
        p = p - n1.dot(p) * n1;
        // Add a vote to a bin
        degreeHist[round(signedAngleBetweenVectors(p,n2,n3))]++;

//        // Compute distance to center
//        double d = radius-p.norm();
//        if(d>0)
//            pointDistHist[std::round(signedAngleBetweenVectors(p,n2,n3))]++;

//        if( (radius-p.norm()) < - 0.5*radius ) N_pp++;
        if( p.norm() < radius*0.75 && np > -0.02 )
        {
            N_pp++;
            iter->r = 0;
            iter->g = 255;
            iter->b = 0;
        }
    }


//    MyHelperFuns::printVector(degreeHist);


    if (N_pp>0)
        std::cout << "Areea point density" << std::pow(radius,2.)*3.14/N_pp << std::endl;

    if ( N_pp>0 && std::pow(radius,2.)*3.14/N_pp < 1E-5 )
    {
        hasOpening = false;
        return;
    }


    // Normalize votes
    double sum = 0;
    for(std::vector<float>::iterator valPtr = degreeHist.begin(); valPtr!=degreeHist.end(); valPtr++)
    {
        sum += *valPtr;
        *valPtr  = *valPtr/cloud->size();
    }


    // Compute entropy
    float entropy = 0.0;
    for(std::vector<float>::iterator valPtr = degreeHist.begin(); valPtr!=degreeHist.end(); valPtr++)
        if(*valPtr>1E-12)
            entropy -= *valPtr * std::log(*valPtr);


    std::cout << "Entropy " << entropy << std::endl;
    std::cout << "In circle " << N_pp << std::endl;

    if( N_pp < 120 && entropy > 5.1 )
    {
        hasOpening = true;
        std::cout << "\033[1;31mFound Opening!\033[0m\n";
    }

}


void
FeatureOpening::circle3DFit()
{
    /* Model: center, radius, normal.*/
    SAC3DCircleWithNormal mySACfit(cloud, cloudNormals, circleInliers);
    Eigen::VectorXf model_coefficients, bestModel;

    double threshold = 0.001;
    int bestFitScore = 0;
    int maxSamples = 200;



    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> dis(0, cloud->size());

    std::vector<int> samples;
    // Do ransac iterations
    V3f nrml, axis(0,-1,0);
    V4f plane, ptPlane, pt;

    for ( int i_iter = 0; i_iter < maxSamples; i_iter++ ){
        int maxbad = 0;
//        printf("Running iteration %d\n",i_iter);

        // Find 3D circle model that holds as an opening
        while(true)
        {
            // If we have done 10 000 samples we can guess that there is no opening
            if(maxbad > 5000)
            {
                bestModel = model_coefficients;
                break;
            }


            // 1. Pick 3 random points in point cloud.
            samples.clear();
            samples.push_back( dis(gen) );
            samples.push_back( dis(gen) );
            samples.push_back( dis(gen) );

            // 2. Fit model
            mySACfit.computeModelCoefficients(samples,model_coefficients);


            // 3. Check max radius and normal alignment
            bool isSampleOk = true;
//            bool isSampleOk = mySACfit.isSampleGood2 (model_coefficients, mainAxis_, objcenter_,maxRadius_);
            if(!isSampleOk)
            {
                maxbad++;
                continue;
            }

            // 4. Check that the 3D circle does not contain too many cloud
            // points and that the inliers are evenly distributed
            std::vector<int> inliers;
            mySACfit.selectWithinDistance (model_coefficients,threshold,inliers);
            bool isModelOk = isCircleOK(model_coefficients,inliers);
            if(!isModelOk)
            {
                maxbad++;
                continue;
            }


            if(isSampleOk && isModelOk)
                break;

            maxbad++;
        }

        if(maxbad > 5000 )
            break;


        std::vector<int> inliers;
        mySACfit.selectWithinDistance (model_coefficients,threshold,inliers);



        /***************************************************/
        // 4. Compute # of inliers to model, i.e., ransac score
        int fitScore = mySACfit.countWithinDistance(model_coefficients,threshold);



        /***************************************************/
        // 4.1 Compute a fit score that is better if the circle is higher up along the
        // the main axis
        nrml(0) = model_coefficients[4];
        nrml(1) = model_coefficients[5];
        nrml(2) = model_coefficients[6];
        nrml.normalize();
        if( nrml(1) > 0 )
        {
            axis(0) = nrml(1)/nrml(0);
            EigenHelperFuns::rotVec180deg(nrml,axis);
        }

        ptPlane(0) = model_coefficients[0];
        ptPlane(1) = model_coefficients[1];
        ptPlane(2) = model_coefficients[2];
        ptPlane(3) = 0;

        plane(0) = nrml(0);
        plane(1) = nrml(1);
        plane(2) = nrml(2);
        plane(3) = -plane.dot(ptPlane);

        double ptsBelowOpening = 0.0;
        for(PC::iterator iter = cloud->begin(); iter!=cloud->end(); ++iter)
        {
            pt = iter->getVector4fMap();
            pt(3)=1;
            pt = pt - ptPlane;
            if(plane.dot(pt)<0)
                ptsBelowOpening++;
        }

        fitScore = (ptsBelowOpening/cloud->size()) *100 + fitScore;



        /***************************************************/
        // 4.2 Compute the alignment of the circle with respect to the inliers
        // Normally the circle normal should be somewhat orthogonal to the inliers of the cirlces point normals

        int nrmlscore = 0;
        for(uint idx = 0; idx < inliers.size(); idx++ )
        {
            V3f ptNrml = cloudNormals->at(inliers[idx]).getNormalVector3fMap();
            nrmlscore += 90-EigenHelperFuns::angleBetweenVectors(nrml, ptNrml,true) < 15 ? 1 : 0;
        }

        fitScore += nrmlscore;



        // 5. Keep model if best sofar!
        if(fitScore > bestFitScore)
        {
//            printf("Normalscore %d \n",nrmlscore);
//            printf("fitScore %d \n",fitScore);
            bestFitScore = fitScore;
            bestModel = model_coefficients;
        }

//        if( (i_iter % 1000) ==0)
//            printf("i_iter %d\n",i_iter);

    }


    model_coefficients = bestModel;
//    printf("Checking model validity\n");


    // Check that the model is valid
    if( model_coefficients.size () != 7 )
        throw std::runtime_error("Found no circle on object!\n");

    // 6. Get the inliers
    mySACfit.selectWithinDistance(model_coefficients,threshold,circleInliers->indices);

    // Init the circle model
    for( uint idx = 0; idx < model_coefficients.size(); idx++ )
    {
        pmCircle->values.push_back(model_coefficients[idx]);
    }

    // HACK
    // Rotate main axis vector 180deg if it is pointing downwards
    V3f circleNormal(pmCircle->values[4],pmCircle->values[5],pmCircle->values[6]);
    if( circleNormal(1) > 0 )
    {
        V3f rotationAxis(-(circleNormal(1)/circleNormal(0)),1.0,0.0);
        Eigen::AngleAxis<float> rotatePIradAroundSubAxis(M_PI,rotationAxis);
        Eigen::Transform<float,3,3> t(rotatePIradAroundSubAxis);
        circleNormal = (t * circleNormal);
        circleNormal.normalize();
        pmCircle->values[4] = circleNormal(0);
        pmCircle->values[5] = circleNormal(1);
        pmCircle->values[6] = circleNormal(2);
    }

}


float
FeatureOpening::signedAngleBetweenVectors(const V3f& v1, const V3f& v2, const V3f& v3)
{
    V3f n1 = v1.normalized();
    V3f n2 = v2.normalized();
    V3f n3 = v3.normalized();

    float dp2 = n1.dot(n3);
    float ang = std::acos( n1.dot(n2) )*(180.0/M_PI);

    if(dp2>0)
        return ang;
    else
        return 360.0-ang;
}


bool
FeatureOpening::isCircleOK(Eigen::VectorXf &model_coefficients, std::vector<int> &inliers)
{

    double r = model_coefficients[3];
    V3f center = V3f(model_coefficients[0],model_coefficients[1],model_coefficients[2]);
    V3f nrml = V3f(model_coefficients[4],model_coefficients[5],model_coefficients[6]);
    nrml.normalize();

    V3f n2(-1,nrml(0)/nrml(1),0);
    n2.normalize();
    V3f n3 = nrml.cross(n2);
    n3.normalize();
    EigenHelperFuns::gmOrth(nrml,n2,n3);

    float radius = model_coefficients[3];
    int N_pp = 0;
    V3f p;
    std::vector<float> degreeHist(360,0);

    for(uint idx = 0; idx < cloud->size(); idx++ )
    {
        // Move point to circle center and remove normal component
//        p = cloud->at(inliers[idx]).getVector3fMap() - center;
        p = cloud->at(idx).getVector3fMap() - center;
        p = p - nrml.dot(p) * nrml;
//        V3f ptNrml = cloudNormals->at(inliers[idx]).getNormalVector3fMap();
        // Add a vote to a bin
//        degreeHist[round(EigenHelperFuns::angleBetweenVectors(nrml,n2,true))]++;
        degreeHist[round(signedAngleBetweenVectors(p,n2,n3))]++;

        double np = nrml.dot(p);
        if( p.norm() < radius*0.75 && np > -0.02 )
        {
            N_pp++;
        }
        if( N_pp > 120)
        {
//            printf("Too many points inside\n");
            return false;

        }
    }

    if ( N_pp>0 && std::pow(r,2.)*3.14/N_pp < 1E-5 )
        return false;

    // Normalize votes
    double sum = 0;
    for(std::vector<float>::iterator valPtr = degreeHist.begin(); valPtr!=degreeHist.end(); valPtr++)
    {
        sum += *valPtr;
        *valPtr  = *valPtr/cloud->size();
    }

    // Compute entropy
    float entropy = 0.0;
    for(std::vector<float>::iterator valPtr = degreeHist.begin(); valPtr!=degreeHist.end(); valPtr++)
        if(*valPtr>1E-12)
            entropy -= *valPtr * std::log(*valPtr);

//    printf("Entropy %f, ",entropy);

    if( N_pp < 120 && entropy > 5.1 )
    {
//          std::cout << "\033[1;31mFound Opening!\033[0m\n";
        return true;

    }
    else{
//        printf("Notok\n");
        return false;}
}


/*
 * Computes: If the pos given is above the opening, the distance to the
 * plane of the circle, and the angle between the circle plane and the pos.
 */
std::vector<double>
FeatureOpening::computePosRelativeOpening ( const V3f &approachVec, const V3f &pos ) const
{
    std::vector<double> posInfo(3,0.0);

    // Compute circle plane
    V4f plane(cNormal(0),cNormal(1),cNormal(2),0);
    plane(3) = -plane.dot(cCenter);

    if( hasOpening )
    {
        V4f v4Pos(pos(0),pos(1),pos(2),1);
//        V4f v4Approach(approachVec(0),approachVec(1),approachVec(2),0);
        // Distance from plane
        posInfo[1] = plane.dot(v4Pos);
        // Above or below opening
        posInfo[0] = (posInfo[1]>0) ? 1 : -1;
        v4Pos(3)=0;
        // Angle with plane
//        posInfo[2] = DEG2RAD(std::acos(cNormal.dot(cCenter-v4Pos)));
        posInfo[2] = EigenHelperFuns::angleBetweenVectors(cNormal.head<3>(),approachVec);
    }

    return posInfo;
}


void
FeatureOpening::generateCloudCircle ( PC::Ptr &cloud )
{

    V3f center = cCenter.head<3>();
    V3f n1 = cNormal.head<3>();
    V3f n2(-1,n1(0)/n1(1),0);
    n2.normalize();
    V3f n3 = n1.cross(n2);
    n3.normalize();
    EigenHelperFuns::gmOrth(n1,n2,n3);

    V3f rotVec ;

//    printf("Radius: %f\n",radius);

    V3f genPT;
    PointT pt;
    for(uint ii = 0; ii!=360; ii++)
    {
        rotVec = n3;
        EigenHelperFuns::rotVecDegAroundAxis(n1,ii,rotVec);
        rotVec.normalize();
        for(int jj = -1; jj!=2; jj++)
        {
            genPT = center + radius * (rotVec) + n1*jj*1E-3;
            pt.x = genPT(0);
            pt.y = genPT(1);
            pt.z = genPT(2);
            pt.r = 0; pt.g = 255; pt.b = 0;
            cloud->push_back(pt);
        }
    }

    // Add center point
    genPT = center;
    pt.x = genPT(0);
    pt.y = genPT(1);
    pt.z = genPT(2);
    pt.r = 0; pt.g = 125; pt.b = 0;
    cloud->push_back(pt);

}


bool
FeatureOpening::isCircleGood ()
{
    double radius = pmCircle->values[3];
    if ( radius > maxRadius_)
    {
//        printf("Radius fail: %f, %f",model_coefficients[3],maxRadius);
//        printf("Radius bad! \n");
        return false;
    }

    // Assign to Eigen representation to write less code..
    V3f cCenter(pmCircle->values[0],pmCircle->values[1],pmCircle->values[2]);
    V3f n(pmCircle->values[4],pmCircle->values[5],pmCircle->values[6]);
    n.normalize();

    // Compute angle between normal and the wanted normal
    double dp = n.dot(mainAxis_);
    double angle_diff = std::acos(dp);
    angle_diff = (std::min) (angle_diff, M_PI - angle_diff);


    V3f voccc = (objcenter_ - cCenter); voccc.normalize();
    double prj  = std::acos(voccc.dot(mainAxis_));
    prj = (std::min) (prj, M_PI - prj);

    std::cout << "Angle diff " << RAD2DEG(prj) << std::endl;
//    std::cout << "Prj diff " << prj << " dp " << dp <<  std::endl;

    // Check whether the current plane model satisfies our angle threshold criterion with respect to the given axis
    if( angle_diff < DEG2RAD(3) && prj < DEG2RAD(20) )
        return (true);
    else
        return (false);
}
