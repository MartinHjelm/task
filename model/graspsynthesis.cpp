#include "graspsynthesis.h"

// STD
#include <iostream>
#include <ctime>

// Boost
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// PCL
#include <pcl/common/transforms.h>

// Mine
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>



GraspSynthesis::GraspSynthesis() :
    cloud_(new PC),
    cloudNormals_(new PCN)
{}


void
GraspSynthesis::setInputSource(const PC::Ptr cloud, const PCN::Ptr cloudNormals, const V4f &plCf)
{
    cloud_ = cloud;
    cloudNormals_ = cloudNormals;
    plCf_ = plCf;
}


bool
GraspSynthesis::sampleGrasp( cuboid &graspBB )
{
    printf("sampleGrasp\n");fflush(stdout);

    //    std::clock_t start;
    int j = 0;

    // Cuboid axis vectors
    V3f n1, n2, n3;

    // Cuboid containers
    Eigen::Matrix3f cuboidAxis,covariance,eigDx;
    Eigen::Matrix4f p3w(Eigen::Matrix4f::Identity());
    V3f mean_diag, pt;
    V4f centroid;
    PointT min_pt, max_pt;
    PC::Ptr cloud(new PC);

    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());

    boost::random::uniform_int_distribution<> distPinchDepth(1,9);
    boost::random::uniform_int_distribution<> distPinchHeight(1,5);
    boost::random::uniform_int_distribution<> distZeroOne(1,2);
    boost::random::uniform_int_distribution<> distOut(2,6);
    boost::random::uniform_int_distribution<> distRotate(0,360);



    while(true)
    {
        j++;
        if(j%100==0)
        {
            //            printf("Slice sampling found nothing after 1000 iters!\n");
            break;
        }

        //        start = std::clock();
        /***** SAMPLE PLANE ******/
        //        pcl::copyPointCloud(*cloudSegmented,*cloud);
        cloud->clear();

        // Sample a plane from the cloud
        V4f plParams(0.0,0.0,0.0,0.0); V3f plPt(0.0,0.0,0.0);
        PCLHelperFuns::samplePlane(  cloud_, plParams, plPt );

        // Get plane normal
        n1 = plParams.head<3>();
        n1.normalize();

        // Rotate normal 2
        Eigen::Vector3f n2(-n1(1),n1(0),0);
        EigenHelperFuns::rotVecDegAroundAxis(n1,distRotate(gen),n2);
        n2.normalize();

        Eigen::Vector3f n3 = n1.cross(n2);
        n3.normalize();

        // Run Gram Schmidt to get it just right...
        EigenHelperFuns::gmOrth(n1,n2,n3);


        //        std::cout << "Time for plane sampling: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();

        /***** CREATE CUBOID ******/


        //        double plMax = 0.0;
        //        PointT cldPt;
        // Project points onto plane
        for(PC::iterator cldIter = cloud_->begin(); cldIter!=cloud_->end(); ++cldIter)
        {
            //            pt = cldIter->getVector3fMap()-plPt;
            //            double projLen = n1.dot(pt) + plParams(3);
            //            if(projLen > plMax)
            //                plMax = projLen;


            if( std::fabs(n2.dot(cldIter->getVector3fMap()) + plParams(3)) < 0.04)
                cloud->push_back(*cldIter);
        }

        if(cloud->size()<100)
            continue;

        // Compute principal directions in the plane
        pcl::compute3DCentroid(*cloud, centroid);
        //        pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
        //        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        //        eigDx = eigen_solver.eigenvectors();
        //        eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));
        // Eigen values are sorted in ascending order...

        // Compute new normals selecting wrist orientation randomly, as either max or min variation
        //        int graspDirection = distZeroOne(gen);
        //        n2 = eigDx.col(graspDirection); n2.normalize();
        //        n3 = n1.cross(n2); n3.normalize();
        //        EigenHelperFuns::gmOrth(n1,n2,n3);

        cuboidAxis.col(0) = n1;
        cuboidAxis.col(1) = n2;
        cuboidAxis.col(2) = n3;


        // 3. Move the points to the new reference frame
        //        cloud->clear();
        pcl::compute3DCentroid(*cloud, centroid);
        p3w.block<3,3>(0,0) = cuboidAxis.transpose();
        p3w.block<3,1>(0,3) = -1.f * (p3w.block<3,3>(0,0) * centroid.head<3>() );
        pcl::transformPointCloud(*cloud, *cloud, p3w);

        // 4. Get min max points
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
        //mean_diag(0) = 0;

        // Set box mid point
        //        graspBB.transVec = cuboidAxis.transpose()*mean_diag;
        graspBB.transVec = cuboidAxis*mean_diag + centroid.head<3>();
        //        graspBB.transVec = centroid.head<3>();
        //        graspBB.transVec = cuboidAxis.transpose()*mean_diag + plPt;
        //        graspBB.transVec(0) = plPt(0); // Interesting change!!
        ////        if(graspDirection==2)
        //            graspBB.transVec(2) = plPt(2);
        ////        else
        //            graspBB.transVec(1) = plPt(1);


        // 6. Transformation values to cuboid
        Eigen::Quaternionf qfinal(cuboidAxis);
        qfinal.normalize();
        graspBB.quartVec = qfinal;
        graspBB.axisMat = cuboidAxis;


        // 7. Generate sides of box

        // For n1 approach vector direction
        graspBB.width = distPinchDepth(gen)*1E-2;
        //        graspBB.width = max_pt.x - min_pt.x;

        // For n2 pinchers height
        graspBB.height = distPinchHeight(gen)*1E-2;
        //        graspBB.height = max_pt.y - min_pt.y;

        // For n3 grasp accross
        //graspBB.depth = max_pt.z - min_pt.z; //dis(gen)*1E-2;//
        graspBB.depth = 0.15;

        // Move center of box out such that the center of the box is at the max point
        int dOut = distOut(gen)*1E-2;
        graspBB.transVec += dOut*n1 + 0.5*(max_pt.x-graspBB.width*0.5)*n1;
        //        graspBB.transVec += 0.5*(max_pt.x + 0.005 - graspBB.width*0.5)*n1;
        graspBB.width = (0.5*graspBB.width + max_pt.x);
        //        graspBB.transVec += 0.5*plMax * n1;
        //        graspBB.width = (0.5*graspBB.width + plMax);

        // Set min/max points
        graspBB.x_min = -0.5f*graspBB.width; graspBB.x_max = 0.5f*graspBB.width;
        graspBB.y_min = -0.5f*graspBB.height; graspBB.y_max = 0.5f*graspBB.height;
        graspBB.z_min = -0.5f* graspBB.depth; graspBB.z_max = 0.5f*graspBB.depth;
        graspBB.setAxisRotMat();

        //        std::cout << "Time for bounding box: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();


        if(!isSampledGraspOK(graspBB))
            continue;


        //        graspPlaneVec.push_back(plParams);

        //        std::cout << "Time for bounding box check: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;


        return true;
    }

    return false;

}



bool
GraspSynthesis::sampleGraspFromCloud( cuboid &graspBB )
{
    printf("sampleGraspFromCloud\n");fflush(stdout);

    // std::clock_t start;
    int j = 1;

    // Cuboid containers
    Eigen::Matrix3f cuboidAxis,covariance,eigDx;
    Eigen::Matrix4f p3w(Eigen::Matrix4f::Identity());
    V3f mean_diag,transVec, pt;
    V4f centroid;
    PointT min_pt, max_pt;

    PC::Ptr cloud(new PC);

    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> dis(0, cloud_->size()-1);
    boost::random::uniform_int_distribution<> distRotate(0,360);
    boost::random::uniform_int_distribution<> distPinchDepth(1,9);
    boost::random::uniform_int_distribution<> distPinchHeight(1,5);
    boost::random::uniform_int_distribution<> distOut(2,6);



    while(true)
    {
        j++;
        if(j%100==0)
        {
            //            printf("Cloud Sampling found nothing after 1000 iters!\n");
            break;
        }


        // start = std::clock();


        /***** SAMPLE POINT IN CLOUD AND NORMAL ******/
        cloud->clear();
        //        pcl::copyPointCloud(*cloud_,*cloud);

        // Sample a plane from the cloud
        V4f plParams(0.0,0.0,0.0,0.0); V3f plPt(0.0,0.0,0.0);
        int pIdx = dis(gen);
        //        printf("Cloud point %i\n",pIdx);
        plPt = cloud_->at(pIdx).getVector3fMap();
        V3f n1 = cloudNormals_->at(pIdx).getNormalVector3fMap();
        //        EigenHelperFuns::rotVecDegAroundAxis(V3f(1,0,0),distRotate(gen),n1);
        //        EigenHelperFuns::rotVecDegAroundAxis(V3f(0,1,0),distRotate(gen),n1);
        //        EigenHelperFuns::rotVecDegAroundAxis(V3f(0,0,1),distRotate(gen),n1);
        n1.normalize();

        //        std::cout << "Time for plane sampling: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();

        // Get normal and create other two normals
        Eigen::Vector3f n2(-n1(1),n1(0),0);
        EigenHelperFuns::rotVecDegAroundAxis(n1,distRotate(gen),n2);
        n2.normalize();
        //        EigenHelperFuns::rotVecDegAroundAxis(n2,distRotate(gen),n1);
        //        n1.normalize();
        Eigen::Vector3f n3 = n1.cross(n2);
        n3.normalize();
        EigenHelperFuns::gmOrth(n1,n2,n3);

        plParams.head<3>() = n1;
        plParams(3) = -n1.dot(plPt);

        //        std::cout << "Time for axis computations: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();


        /***** CREATE CUBOID ******/
        //        double plMax = 0.0;
        //        PointT cldPt;
        // Project points onto plane
        for(PC::iterator cldIter = cloud_->begin(); cldIter!=cloud_->end(); ++cldIter)
        {
            //            pt = cldIter->getVector3fMap()-plPt;
            //            double projLen = n1.dot(pt) + plParams(3);
            //            if(projLen > plMax)
            //                plMax = projLen;


            if( std::fabs(n2.dot(cldIter->getVector3fMap()) + plParams(3)) < 0.04)
                cloud->push_back(*cldIter);
        }

        if(cloud->size()<100)
            continue;

        // Compute principal directions in the plane
        cuboidAxis.col(0) = n1;
        cuboidAxis.col(1) = n2;
        cuboidAxis.col(2) = n3;


        // 3. Move the points to the new reference frame
        pcl::compute3DCentroid(*cloud, centroid);
        p3w.block<3,3>(0,0) = cuboidAxis.transpose();
        p3w.block<3,1>(0,3) = -1.f * (p3w.block<3,3>(0,0) * centroid.head<3>() );
        pcl::transformPointCloud(*cloud, *cloud, p3w);

        // 4. Get min max points
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
        //mean_diag(0) = 0;

        // Set box mid point
        //        graspBB.transVec = plPt;
        //        graspBB.transVec = cuboidAxis.transpose()*mean_diag;
        //        graspBB.transVec = cuboidAxis.transpose()*mean_diag + centroid.head<3>();
        graspBB.transVec = centroid.head<3>();
        //        graspBB.transVec = cuboidAxis.transpose()*mean_diag + plPt;
        //        graspBB.transVec(0) = plPt(0); // Interesting change!!
        ////        if(graspDirection==2)
        //            graspBB.transVec(2) = plPt(2);
        ////        else
        //            graspBB.transVec(1) = plPt(1);


        // 6. Transformation values to cuboid
        Eigen::Quaternionf qfinal(cuboidAxis);
        qfinal.normalize();
        graspBB.quartVec = qfinal;
        graspBB.axisMat = cuboidAxis;


        // 7. Generate sides of box

        // For n1 approach vector direction
        graspBB.width = distPinchDepth(gen)*1E-2;
        //        graspBB.width = max_pt.x - min_pt.x;

        // For n2 pinchers height
        graspBB.height = distPinchHeight(gen)*1E-2;
        //        graspBB.height = max_pt.y - min_pt.y;

        // For n3 grasp accross
        graspBB.depth = max_pt.z - min_pt.z; //dis(gen)*1E-2;//
        //        graspBB.depth = 0.15;
        //        graspBB.depth = distPinchDepth(gen)*1E-2;

        // Move center of box out such that the center of the box is at the max point
        int dOut = distOut(gen)*1E-2;
        graspBB.transVec += dOut*n1 + 0.5*(max_pt.x-graspBB.width*0.5)*n1;
        //        graspBB.transVec += 0.5*(max_pt.x + 0.005 - graspBB.width*0.5)*n1;
        graspBB.width = (0.5*graspBB.width + max_pt.x);
        //        graspBB.transVec += 0.5*plMax * n1;
        //        graspBB.width = (0.5*graspBB.width + plMax);

        // Set min/max points
        graspBB.x_min = -0.5f*graspBB.width; graspBB.x_max = 0.5f*graspBB.width;
        graspBB.y_min = -0.5f*graspBB.height; graspBB.y_max = 0.5f*graspBB.height;
        graspBB.z_min = -0.5f* graspBB.depth; graspBB.z_max = 0.5f*graspBB.depth;
        graspBB.setAxisRotMat();

        //        std::cout << "Time for bounding box: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();


        if(!isSampledGraspOK(graspBB))
            continue;


        //        graspPlaneVec.push_back(plParams);


        return true;
    }

    return false;

}




/* Samples grasp by finding a plane in the pointcloud. Projecting all points in
the cloud down on the plane. Computing the princioal components and then placing
a grasp that has an approach vector to the plane. The widht of the gripper is
the min of the two principal components.*/
bool
GraspSynthesis::sampleGraspCentroid( cuboid &graspBB )
{
    printf("sampleGraspCentroid\n");fflush(stdout);

    //    std::clock_t start;
    int j = 0;

    // Cuboid axis vectors
    V3f n1, n2, n3;

    // Cuboid containers
    Eigen::Matrix3f cuboidAxis,covariance,eigDx;
    Eigen::Matrix4f p3w(Eigen::Matrix4f::Identity());
    V3f mean_diag, pt;
    V4f centroid;
    PointT min_pt, max_pt;

    PC::Ptr cloudCopy(new PC);

    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> distPinchDepth(1,9);
    boost::random::uniform_int_distribution<> distPinchHeight(2,10);
    boost::random::uniform_int_distribution<> distZeroOne(1,2);
    boost::random::uniform_int_distribution<> distOut(2,6);


    while(true)
    {
        j++;
        if(j%100==0)
        {
            //            printf("Centroid Sampling found nothing after 1000 iters!\n");
            break;
        }


        //        start = std::clock();



        /** SAMPLE PLANE **/
        pcl::copyPointCloud(*cloud_,*cloudCopy);

        // Sample a plane from the cloud
        V4f plParams(0.0,0.0,0.0,0.0); V3f plPt(0.0,0.0,0.0);
        PCLHelperFuns::samplePlane(cloudCopy, plParams, plPt );

        // Get plane normal
        n1 = plParams.head<3>();
        n1.normalize();


        //        std::cout << "Time for plane sampling: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();



        /** CREATE CUBOID **/

        // Project points onto plane
        for(PC::iterator cldIter = cloudCopy->begin(); cldIter!=cloudCopy->end(); ++cldIter)
            cldIter->getVector3fMap() = pt - n1 * n1.dot(pt);

        // Compute principal directions in the plane
        pcl::compute3DCentroid(*cloudCopy, centroid);
        pcl::computeCovarianceMatrixNormalized(*cloudCopy, centroid, covariance);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        eigDx = eigen_solver.eigenvectors();
        eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));
        // Eigen values are sorted in ascending order...

        // std::cout << "The eigenvalues of A are:" << std::endl << eigen_solver.eigenvalues() << std::endl;

        // Compute new normals selecting wrist orientation randomly, as either max or min variation
        int graspDirection = distZeroOne(gen);
        n2 = eigDx.col(graspDirection); n2.normalize();
        n3 = n1.cross(n2); n3.normalize();
        EigenHelperFuns::gmOrth(n1,n2,n3);

        cuboidAxis.col(0) = n1;
        cuboidAxis.col(1) = n2;
        cuboidAxis.col(2) = n3;


        // 3. Move the points to the new reference frame
        cloudCopy->clear();
        pcl::compute3DCentroid(*cloud_, centroid);
        p3w.block<3,3>(0,0) = cuboidAxis.transpose();
        p3w.block<3,1>(0,3) = -1.f * (p3w.block<3,3>(0,0) * centroid.head<3>() );
        pcl::transformPointCloud(*cloud_, *cloudCopy, p3w);

        // 4. Get min max points
        pcl::getMinMax3D(*cloudCopy, min_pt, max_pt);
        mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
        mean_diag(0) = 0;

        // Set box mid point
        graspBB.transVec = cuboidAxis.transpose()*mean_diag + centroid.head<3>();
        //        graspBB.transVec = centroid.head<3>();


        // 6. Transformation values to cuboid
        Eigen::Quaternionf qfinal(cuboidAxis);
        qfinal.normalize();
        graspBB.quartVec = qfinal;
        graspBB.axisMat = cuboidAxis;


        // 7. Generate sides of box
        // For n1 approach vector direction
        graspBB.width = distPinchDepth(gen)*1E-2;
        //        graspBB.width = (max_pt.x - min_pt.x);

        // For n2 pinchers height
        graspBB.height = distPinchHeight(gen)*1E-2;
        //graspBB.height = max_pt.y - min_pt.y;

        // For n3 grasp accross
        graspBB.depth = (max_pt.z - min_pt.z);


        // Move center out in grasp direction random distance
        int dOut = distOut(gen)*1E-2;
        graspBB.transVec += dOut*n1 + 0.5*(max_pt.x-graspBB.width*0.5)*n1;
        graspBB.width = (0.5*graspBB.width + max_pt.x);

        //        graspBB.transVec += dOut*n1 + 0.5*plMax * n1;
        //        graspBB.width = (0.5*graspBB.width + plMax);


        // Set min/max points
        graspBB.x_min = -0.5f*graspBB.width; graspBB.x_max = 0.5f*graspBB.width;
        graspBB.y_min = -0.5f*graspBB.height; graspBB.y_max = 0.5f*graspBB.height;
        graspBB.z_min = -0.5f*graspBB.depth; graspBB.z_max = 0.5f*graspBB.depth;
        graspBB.setAxisRotMat();

        //        std::cout << "Time for bounding box: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();

        if(!isSampledGraspOK(graspBB))
            continue;

        //        graspPlaneVec.push_back(plParams);

        //        std::cout << "Time for bounding box check: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;


        return true;
    }

    return false;

}



bool
GraspSynthesis::sampleGraspPoints( cuboid &graspBB )
{
//    printf("sampleGraspPoints\n"); fflush(stdout);

    std::clock_t start;
    int j = 1;

    // Cuboid containers
    Eigen::Matrix3f cuboidAxis;

    PC::Ptr cloud(new PC);
    boost::random::random_device rd;
    boost::random::mt19937 gen(rd());
    boost::random::uniform_int_distribution<> distPC(0, cloud_->size()-1);
    boost::random::uniform_int_distribution<> distRotate(0,180);
    boost::random::uniform_int_distribution<> distPinchDepth(1,9);
    boost::random::uniform_int_distribution<> distPinchHeight(1,5);
    boost::random::uniform_int_distribution<> distOut(2,6);



    while(true)
    {
        j++;
        if(j%100==0)
        {
            //            printf("Cloud Sampling found nothing after 1000 iters!\n");
            break;
        }

        // Reset variables
        // start = std::clock();
        cloud->clear();


        // Sample point from the cloud
        int ptIdx = distPC(gen);
        V3f pt = cloud_->at(ptIdx).getVector3fMap();

        // Create cuboid axes (n1 is the approach vector)
        V3f n1 = cloudNormals_->at(ptIdx).getNormalVector3fMap();
        V3f n2(-n1(1),n1(0),0);
        V3f n3 = n1.cross(n2);

        // Create random rotation around each of the axes
        double angle = distRotate(gen);
        EigenHelperFuns::rotVecDegAroundAxis(n1,angle,n2);
        EigenHelperFuns::rotVecDegAroundAxis(n1,angle,n3);
        angle = distRotate(gen);
        EigenHelperFuns::rotVecDegAroundAxis(n2,angle,n1);
        EigenHelperFuns::rotVecDegAroundAxis(n2,angle,n3);
        angle = distRotate(gen);
        EigenHelperFuns::rotVecDegAroundAxis(n3,angle,n1);
        EigenHelperFuns::rotVecDegAroundAxis(n3,angle,n2);

        // Fix basis
        EigenHelperFuns::gmOrth(n1,n2,n3);

        // Let the plane that we are grasping orthogonally have sides 2cm
        // and check that the there is enough room to place the grasp that is
        // 2cm in either direction is free
        double cubeSideLen = 0.02;
        int n1Counter = 0;
        int n2Counter = 0;
        int n3Counter = 0;
        double cDiag = std::sqrt(3) * (cubeSideLen+0.02);
        bool freeApproach = true;
        for(PC::const_iterator ptPtr = cloud_->begin(); ptPtr!=cloud_->end(); ++ptPtr)
        {
            V3f cldPt = ptPtr->getVector3fMap()-pt;

            double len1 = n1.dot(cldPt);
            double len2 = std::fabs(n2.dot(cldPt));
            double len3 = std::fabs(n3.dot(cldPt));

//            printf("%f,%f,%f\n",len1,len2,len3);

            // Check that the approach vector is free
            if(len1>0.01 && len2<cubeSideLen+0.02 && len3<cubeSideLen+0.02)
            {
                freeApproach = false;
                break;
            }

            if(cldPt.norm()>cDiag)
                continue;

            // Project down onto all axes
            len1 = std::fabs(n1.dot(cldPt));

            if(len2>cubeSideLen && len2<cubeSideLen+0.04 && len1 < cubeSideLen )
            {
                n2Counter++;
            }


            if(len3>cubeSideLen && len3<cubeSideLen+0.02 && len1 < cubeSideLen )
            {
                n3Counter++;
            }
        }

        if(!freeApproach)
            continue;

//        if(n2Counter==0)
//            printf("C1 %i, C2 %i, C3 %i\n",n1Counter,n2Counter,n3Counter); fflush(stdout);

        if(n1Counter!=0 || (n2Counter>0 && n3Counter>0) )
            continue;
//            return false;


        // K seems like we can grasp this cuboid

        /***** CREATE CUBOID ******/

        // Compute principal directions in the plane
        cuboidAxis.col(0) = n1;
        cuboidAxis.col(1) = n2;
        cuboidAxis.col(2) = n3;
        graspBB.transVec = pt;
        Eigen::Quaternionf qfinal(cuboidAxis);
        qfinal.normalize();
        graspBB.quartVec = qfinal;
        graspBB.axisMat = cuboidAxis;

        // For approach vector direction
        graspBB.width = 0.02; //distPinchDepth(gen)*1E-2;
        // For n2 pinchers height
        graspBB.height = 0.04; //distPinchHeight(gen)*1E-2;
        // For n3 grasp accross
        graspBB.depth = 0.02;

        // Move center of box out such that the center of the box is at the max point
//        int dOut = distOut(gen)*1E-2;
//        graspBB.transVec += dOut*n1 + 0.5*(0.02-graspBB.width*0.5)*n1;

        // Set min/max points
        graspBB.x_min = -0.5f*graspBB.width; graspBB.x_max = 0.5f*graspBB.width;
        graspBB.y_min = -0.5f*graspBB.height; graspBB.y_max = 0.5f*graspBB.height;
        graspBB.z_min = -0.5f* graspBB.depth; graspBB.z_max = 0.5f*graspBB.depth;
        graspBB.setAxisRotMat();

        //        std::cout << "Time for bounding box: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //        start = std::clock();

        if(!isSampledGraspOK(graspBB))
            continue;

        return true;
    }

    return false;

}










/** Checks for grasp cuboid if is ok. **/
bool
GraspSynthesis::isSampledGraspOK( cuboid &graspBB )
{
    //    double ang = EigenHelperFuns::angleBetweenVectors( graspBB.axisMat.row(2),plCf_.head<3>());
    //    if( ang < 70 )
    //    {
    //        printf("Angle to small. \n");
    //        return false;
    //    }

    // Check if approach vector is below 1cm above the table.
    if( (graspBB.transVec.transpose()+graspBB.axisMat.row(0)*(graspBB.width*0.5+0.2)).dot(plCf_.head<3>())+plCf_(3) < 0.01)
    {
        //        printf("Approach vector below table. \n");
        return false;
    }

    // Check that grasp across is not too big
    if(graspBB.depth > 0.16 || graspBB.depth < 0.01 )
    {
        //        printf("Grasp accross too big or too small. \n");
        return false;
    }

    // Check if any of the corners of the cube are below 1cm above the table.
    Eigen::MatrixXf cornerMat(8,3);
    graspBB.getCornersOfCuboid( cornerMat );
    for(int iSide=0; iSide!=8; iSide++)
    {
        //        std::cout <<  cornerMat.row(iSide).dot(plCf_.head<3>())+plCf(3)  << std::endl;
        if( cornerMat.row(iSide).dot(plCf_.head<3>())+plCf_(3) < 0.01)
        {
            //            printf("Corners of the graspsing cube are below the table \n");
            return false;
        }
    }

    // Check if there are any points inside the pinch grasp
    pcl::PointIndices::Ptr graspedPtsIdxs(new pcl::PointIndices);

    // We check that the actual gripper contains any points by setting pinch height to 2 cm.
    //    double tmpHeight = graspBB.height;
    //    graspBB.setHeight(0.01);

    PCLHelperFuns::computePointsInsideBoundingBox(cloud_,graspBB,graspedPtsIdxs);
    if(graspedPtsIdxs->indices.size()<50)
    {
        //        printf("Too few points inside the gripper plane. \n");
        return false;
    }

    //        graspBB.setHeight(tmpHeight);

    if(!GraspSynthesis::arePointsInsideBoundingBoxEvenlyDistributed(cloud_,graspBB))
    {
        //        printf("Points inside bounding box not evenly distributed. \n");
        return false;
    }


    //    std::cout << graspedPtsIdxs->indices.size() << std::endl;




    //    graspBB.height = tmpHeight;

    // All checks OK!
    return true;
}



/** Checks if there are points in both sides of the cuboid **/
bool
GraspSynthesis::arePointsInsideBoundingBoxEvenlyDistributed (const PC::Ptr &cloud, const cuboid &bb)
{
    // n2 is the ortogonal

    float Nn2left = 0;
    float Nn2right = 0;
    float Nn3left = 0;
    float Nn3right = 0;


    for(uint iPt = 0; iPt!=cloud->size(); ++iPt)
    {
        if( bb.isPtInCuboid(cloud->points[iPt].getVector3fMap()) )
        {
            V3f ptProj = bb.axisRotMat * (cloud->points[iPt].getVector3fMap()-bb.transVec);
            if(ptProj(1)>0)
                Nn2left++;
            else
                Nn2right++;

            if(ptProj(2)>0)
                Nn3left++;
            else
                Nn3right++;
        }
    }

    // std::cout << Nleft << " " << Nright << std::endl;

    if(Nn2left==0 || Nn2right==0 || Nn3left==0 || Nn3right==0 )
        return false;

    double ratioN2 = Nn2left/Nn2right;
    double ratioN3 = Nn3left/Nn3right;

    if( ratioN2 > 0.7 || ratioN2 < 0.3 || ratioN3 > 0.8 || ratioN3 < 0.2)
        return false;

    return true;
}
