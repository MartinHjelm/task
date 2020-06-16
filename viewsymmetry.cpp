// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//  PCL
#include <PCLTypedefs.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>

// Mine
#include <scenesegmentation.h>
#include <mainaxes.h>
#include <featurecolorquantization.h>
#include <featureopening.h>
#include <featurefpfh.h>
#include <featurecolorhist.h>
#include <myhelperfuns.h>
#include <boosthelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>

double symmetryScore(PC::Ptr &cloud, PCN::Ptr cloudNormals, V3f &midPoint, V3f &n );


/* MAIN */
int
main(int argc, char **argv)
{
    if(argc<2)
    {
        std::cout << "Specify file name. Please!" << std::endl;
        return 0;
    }

    std::string fName = argv[1];
    std::string featureName = "";
    if(argc > 2) featureName = argv[2];

    if(!BoostHelperFuns::fileExist(fName))
    {
        std::cout << "Point cloud file not found!" << std::endl;
        return 0;
    }

    std::clock_t startTime; // Timer


    /************************ POINT CLOUD VIEWER ************************/
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (1.0, 1.0, 1.0);
    viewer->initCameraParameters ();
    viewer->setCameraPosition(0,0,0,0,-1,1,0);
    std::vector<pcl::visualization::Camera> cam;
    viewer->getCameras(cam);
//    viewer->addCoordinateSystem (1.0,);

    printf("Camera parameters: \n");
    std::cout   << " - pos:\t ("  << cam[0].pos[0]    << ", " << cam[0].pos[1]    << ", " << cam[0].pos[2]    << ")"  << std::endl;



    /************************ SCENE SEGMENTATION ************************/
    startTime = std::clock();
    SceneSegmentation SS;
    SS.setInputSource(fName);
    // Segmentation of 3D Scene
    SS.segmentPointCloud();
    //Extract point cloud segmentation window into image to run F's 2D segmentation algorithm
    SS.segmentImage();
    std::cout << "Scene segmentation done." << std::endl;
    std::cout << "Scene Segmentation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;



    /************************ FEATURE COMPUTATIONS ************************/
    startTime = std::clock();
    std::cout << "-------------" << std::endl;
    std::cout << "Computing Main Axes feature" << std::endl;
    // Compute Main Axes Feature
    MainAxes FMA;
    FMA.setInputSource(SS.cloudSegmented, SS.cloudSegmentedNormals,SS.plCf.head<3>());
    FMA.fitObject2Primitives();
    EigenHelperFuns::printEigenVec(FMA.axesLengths,"Axises lengths");
    std::cout << "Main Axis Computation Time: " << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds." << std::endl;





    /************************ SYMMETRY COMPUTATIONS ************************/

    printf("Starting symmetry computations\n");
    printf("------------------------------------------\n");std::fflush(stdout);

    pcl::PointIndices::Ptr inliers;
    switch (FMA.objectPrimitive)
    {
    case 0: //"cylinder"
        inliers = FMA.cylinderInliers;
        break;
    case 1: //"sphere"
        inliers = FMA.sphereInliers;
        break;
    case 2: //"cuboid"
        inliers = FMA.cubeInliers;
        break;
    }

    // Extract inliers from the segmented cloud
    PC::Ptr cloud(new PC);
    PCN::Ptr cloudNormals(new PCN);

//    pcl::ExtractIndices<PointT> extract;
//    extract.setNegative (false);
//    extract.setInputCloud (SS.cloudSegmented);
//    extract.setIndices (inliers);
//    extract.filter (*cloud);

//    pcl::ExtractIndices<PointN> extract_normals;
//    extract_normals.setInputCloud (SS.cloudSegmentedNormals);
//    extract_normals.setIndices (inliers);
//    extract_normals.filter (*cloudNormals);

    pcl::copyPointCloud(*SS.cloudSegmented,*cloud);
    *cloudNormals = *SS.cloudSegmentedNormals;

    PCLHelperFuns::colorIndices(SS.cloudSegmented, inliers, V3f(0,0,255) );



    // Compute reflection along the symmetry plane according to Jeannet's algorithm

    // 1. Select relfection plane and normal as the object vector pointing away from the object
    //    V3f vp(0,0,1);
    V3f vp(cam[0].focal[0]-cam[0].pos[0],cam[0].focal[1]-cam[0].pos[1],cam[0].focal[2]-cam[0].pos[2]);
    vp.normalize();
    EigenHelperFuns::printEigenVec(vp,"Viewing Direction:");

    V3f planeNormal;
    V3f axisrot;

    std::vector<double> sp;
    sp.push_back( std::fabs( vp.dot(FMA.axesVectors.col(1))) );
    sp.push_back( std::fabs( vp.dot(-FMA.axesVectors.col(1))) );
    sp.push_back( std::fabs( vp.dot(FMA.axesVectors.col(2))) );
    sp.push_back( std::fabs( vp.dot(-FMA.axesVectors.col(2))) );

    switch ( std::distance(sp.begin(), std::max_element(sp.begin(), sp.end())) )
    {
    case 0:
        planeNormal = FMA.axesVectors.col(1);
        axisrot = FMA.axesVectors.col(0);
        printf("Picked case 1 for plane.\n");
        break;
    case 1:
        planeNormal = -FMA.axesVectors.col(1);
        axisrot = FMA.axesVectors.col(0);
        printf("Picked case 2 for plane.\n");
        break;
    case 2:
        planeNormal = FMA.axesVectors.col(2);
        axisrot = FMA.axesVectors.col(0);
        printf("Picked case 3 for plane.\n");
        break;
    case 3:
        planeNormal = -FMA.axesVectors.col(2);
        axisrot = FMA.axesVectors.col(0);
        printf("Picked case 4 for plane.\n");
        break;
    }

    planeNormal.normalize();
    axisrot.normalize();

    double bestTrans = 0.0;
    double bestRot = 0.0;
    double bestAxisRot = 0.0;
    double bestScore = 1E10;

    double rotRange = 1;
    double transRange = 1;
    double degree = 5.0;
    double axisdegree = 5.0;
    double displacement = 0.01;
    bool letsRun = true;
    int counter = 0;
    // For every reflection plane roation and translation compute score

//    while(letsRun)
    while(counter<50)
    {

        std::vector<double> scores;
        std::vector<std::vector<double> > mods;

        for(double i_rotAxis=-rotRange; i_rotAxis<=rotRange; i_rotAxis++)
        {
        for(double i_trans=-transRange; i_trans<=transRange; i_trans++)
        {
            for(double i_rot=-rotRange; i_rot<=rotRange; i_rot++)
            {

                double translation = i_trans*displacement+bestTrans;
                double rotation = bestRot+i_rot*degree;
                double axisRotation = bestAxisRot+i_rotAxis*axisdegree;

                printf("Translation: %f Rotation: %f AxisRotation: %f ",translation,rotation,axisRotation);

                // Translation of midpoint
                V3f midPoint = FMA.midPoint + translation*planeNormal;
                V3f n = planeNormal;

                // Plane rotation
//                EigenHelperFuns::rotVecDegAroundAxis(n,axisRotation,axisrot);
                EigenHelperFuns::rotVecDegAroundAxis(axisrot,rotation,n);


                // Compute score
                scores.push_back(symmetryScore(cloud,cloudNormals,midPoint,n));
                std::vector<double> transform;
                transform.push_back(translation);
                transform.push_back(rotation);
                transform.push_back(axisRotation);
                mods.push_back(transform);
            }
        }
        }

        int minIdx = MyHelperFuns::minIdxOfVector(scores);
        if(scores[minIdx] < bestScore )
        {
            bestTrans = mods[minIdx][0];;
            bestRot = mods[minIdx][1];
            bestAxisRot = mods[minIdx][2];
            bestScore = scores[minIdx];
        }
        else if(std::fabs(degree-.5)>1E-6)
        {
            printf("Decreasing degree direction.\n");
            degree-=0.5;
        }
        else if(std::fabs(axisdegree-.5)>1E-6)
        {
            printf("Decreasing degree direction.\n");
            axisdegree-=0.5;
        }
        else if(std::fabs(displacement-0.005)>1E-6)
        {
            printf("Decreasing translation direction.\n");
            displacement -= 0.005;
        }
        else
            break;

        printf("Best score: %f\n",bestScore);

        counter++;
    }

    printf("Best transform: %f\n",bestTrans);
    printf("Best rotation: %f\n",bestRot);

    //    bestTrans = 0.02;
    //    bestRot = -20.0;

    // Compute best rotation and translation
    // Translation of midpoint
    V3f midPoint = FMA.midPoint + bestTrans*planeNormal;
    // Plane rotation
    V3f n = planeNormal;
    EigenHelperFuns::rotVecDegAroundAxis(axisrot,bestRot,n);
    EigenHelperFuns::rotVecDegAroundAxis(n,bestAxisRot,axisrot);
    double b = midPoint.dot(n);

    // 2. Compute reflection
    PC::Ptr cloudQ(new PC);
    for( int idx = 0; idx != cloud->size(); idx++ )
    {
        V3f p = cloud->at(idx).getVector3fMap();
        V3f q = p - 2 * n * ( p.dot(n) - b);
        PointT pt;
        pt = cloud->at(idx);
        pt.getVector3fMap() = q;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        pt.a = 255;

        int row, col;
        PCLHelperFuns::projectPt2Img(pt,row,col);
        if(PCLHelperFuns::isPxInCloud(cloud,row*640+col,p))
        {
            //            printf("In cloud!");
            //            if( std::fabs(q.dot(vp)) > std::fabs(p.dot(vp)) ) // Add point only if it is hidden
            if( q.norm() > p.norm() )
            {
                pt.ID = row*640+col;
                cloudQ->push_back(pt);
                //                printf("reflected!");
            }
        }
        else
        {

            //                        cloudQ->push_back(pt);
        }
    }






    /************************ ADD VIEW PORTS ************************/
    //Add 3 view ports
    int v1(0), v2(0), v3(0);
    // Viewport 1
    viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(SS.cloudScene);
    viewer->addPointCloud<PointT> (SS.cloudScene, rgb1, "Whole scene",v1);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,"Whole scene");
    //    viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

    // Viewport 2
    viewer->createViewPort (0.5, 0.0, 1., 1.0, v2);
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(SS.cloudSegmented);
    viewer->addPointCloud<PointT> (SS.cloudSegmented, rgb2, "Object",v2);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,"Object");


    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbC(cloudQ);
    viewer->addPointCloud<PointT> (cloudQ, rgbC, "Circle1",v2);
    viewer->addPointCloud<PointT> (cloudQ, rgbC, "Circle2",v1);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,"Circle2");

    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0,"Object");
    viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

    //    // Viewport 3
    //    viewer->createViewPort (0.66, 0.0, 1.0, 1.0, v3);
    ////    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(SS.cloudSegmented);
    ////    viewer->addPointCloud<PointT> (SS.cloudSegmented, rgb3, "SObject",v3);
    ////    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"SObject");




    //  ROBOT ORIGIN
    // 566.11077880859375 1227.518310546875 303.9049072265625 -0.18138060909146622 0.20938222550806626 0.76739448403667976 -0.57824377571831764
    //V3f robCoordinate(0.56611077880859375, 0.1227518310546875, 0.03039049072265625);


    //    //  Add object midpoint axes to viewer
    //    Eigen::Vector3f pEnd;
    //    Eigen::Vector3f pStart = FMA.midPoint;
    //    for(int idx=0; idx!=3; idx++)
    //    {
    //        Eigen::Vector3f pTmp = FMA.axesVectors.col(idx);
    //        pEnd = pStart + pTmp * FMA.axesLengths(idx);
    //        PointT p1, p2;
    //        PCLHelperFuns::convEigen2PCL(pStart,p1);
    //        PCLHelperFuns::convEigen2PCL(pEnd,p2);
    //        viewer->addArrow(p2,p1, idx*0.33, 0.5, 0, 0, MyHelperFuns::toString(idx),v2);
    //    }


    // Add symmetry plane axes
    PointT p1, p2;
    Eigen::Vector3f pStart = midPoint;
    Eigen::Vector3f pEnd;
    pEnd = pStart + n * 0.1;
    PCLHelperFuns::convEigen2PCL(pStart,p1);
    PCLHelperFuns::convEigen2PCL(pEnd,p2);

    // Symmetry plane normal
    viewer->addArrow(p2,p1,1, 0, 0, 0, MyHelperFuns::toString(1),v2);
    //    viewer->addText3D ("Symmetry Plane Normal",p2,.1,1.0,0,0,"teststring",v2);

    pEnd = pStart + axisrot * 0.1;
    PCLHelperFuns::convEigen2PCL(pStart,p1);
    PCLHelperFuns::convEigen2PCL(pEnd,p2);
    // Rotation axis
    viewer->addArrow(p2,p1,0, 1.0, 0, 0, MyHelperFuns::toString(2),v2);





    //  Add geometric primitives to the viewer
    switch (FMA.objectPrimitive)
    {
    case 0: //"cylinder"
        viewer->addCylinder(*FMA.pmCylinder,"Cylinder",v2);
        break;
    case 1: //"sphere"
        viewer->addSphere(*FMA.pmSphere,"Sphere",v2);
        break;
    case 2: //"cuboid"
        viewer->addCube(FMA.pmCube_.transVec,FMA.pmCube_.quartVec,FMA.pmCube_.width,FMA.pmCube_.height,FMA.pmCube_.depth,"Cube0",v2);
        // Corners of cuboid
        PC::Ptr cl(new PC);
        FMA.pmCube_.getCornersOfCuboidAsPC(cl);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgbcl(cl);
        viewer->addPointCloud<PointT> (cl, rgbcl, "corners",v2);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,4,"corners");
        break;
    }


    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (10000));
    }

    return 1;

}



double symmetryScore(PC::Ptr &cloud, PCN::Ptr cloudNormals, V3f &midPoint, V3f &n)
{

    // 2. Compute reflection
    double b = midPoint.dot(n);
    double E1 = 0;
    int E1counter = 0;
    double E2 = 0;
    int E2counter = 0;
    double E3 = 0;
    int E3counter = 0;

    PC::Ptr rCloud(new PC);
    PCN::Ptr rCloudNormals(new PCN);
    for( int idx = 0; idx != cloud->size(); idx++ )
    {
        V3f p = cloud->at(idx).getVector3fMap();
        V3f q = p - 2 * n * ( p.dot(n) - b);
        PointT pt;
        pt = cloud->at(idx);
        pt.getVector3fMap() = q;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        pt.a = 255;
        rCloud->push_back(pt);

        //                // 3. Project onto image coordinates
        int row, col;
        PCLHelperFuns::projectPt2Img(pt,row,col);

        //                // 4. Compute scores
        if(PCLHelperFuns::isPxInCloud(cloud,row*640+col,p))
        {
            //                    if( std::fabs(q.dot(vp)) < std::fabs(p.dot(vp)) )
            if( q.norm() < p.norm() )
            {
                //                        E1 =  std::fabs(p.dot(vp)) - std::fabs(q.dot(vp));
                E1 += (q-p).norm();
                E1counter++;
            }

        }
        else
        {
            double dBest = 1E10;
            for( int idx = 0; idx != cloud->size(); idx++ )
            {
                int y = cloud->at(idx).ID/640;
                int x = cloud->at(idx).ID%640;

                // Euclidian
                //                        double d = std::sqrt( std::pow((double)(y-row),2) + std::pow((double)(x-col),2) );
                // City block
                double d = std::fabs((double)(y-row)) + std::fabs((double)(x-col));

                if(d<dBest)
                    dBest = d;
            }
            E2 += dBest;
            E2counter++;
        }
    }


    // Compare cloud normals
    PCLHelperFuns::computeCloudNormals(rCloud,0.01,rCloudNormals);
    for( int id = 0; id != rCloudNormals->size(); id++ )
    {
        V3f pNormal = cloudNormals->at(id).getNormalVector3fMap();
        V3f qNormal = rCloudNormals->at(id).getNormalVector3fMap();
        double angle = std::acos(pNormal.dot(qNormal)/(pNormal.norm()*qNormal.norm()));
        angle = (std::min) (angle, M_PI - angle);
        angle = RAD2DEG(angle);
        if(angle > 1.0 )
        {
            E3 += angle;
            E3counter++;
        }

    }


    // Compare average distance to plane
    double dC = 0.0;
    double dR = 0.0;
    double dCcount = 0.0;
    double dRcount = 0.0;
    std::vector<double> dVec1, dVec2, hist1, hist2;

    for( int id = 0; id != cloud->size(); id++ )
    {
        V3f p = cloud->at(id).getVector3fMap();
        double dp = n.dot(p-midPoint);
        if(dp>0)
        {
            dR += std::fabs(dp);
            dRcount++;
            dVec1.push_back(std::fabs(dp));
        }
        else
        {
            dC += std::fabs(dp);
            dCcount++;
            dVec2.push_back(std::fabs(dp));
        }
    }

    //            for( int id = 0; id != rCloud->size(); id++ )
    //            {
    //                V3f p = rCloud->at(id).getVector3fMap();
    //                double dp = n.dot(p-midPoint);
    //                if(dp>0)
    //                {
    //                    dR += std::fabs(dp);
    //                    dRcount++;
    //                    dVec1.push_back(std::fabs(dp));
    //                }
    //                else
    //                {
    //                    dC += std::fabs(dp);
    //                    dCcount++;
    //                    dVec2.push_back(std::fabs(dp));
    //                }
    //            }

    if(dRcount>0)
        dR /= dRcount;
    if(dCcount>0)
        dC /= dCcount;

    MyHelperFuns::vec2Hist(dVec1,99,hist1);
    MyHelperFuns::vec2Hist(dVec2,99,hist2);

    //            MyHelperFuns::printVector(hist1,"Histogram 1 ");
    //            MyHelperFuns::printVector(hist2,"Histogram 2 ");

    double dHist = MyHelperFuns::manhattanDist(hist1,hist2);

    //            double E4 =0.0;// std::fabs(dR-dC);
    //            double E4 = std::fabs(dRcount-dCcount);
    double E4 = 1E0*dHist;
    double E4counter = 1;




    //            // 4. Compute score
    double score = 0.0;
    if(E1counter > 0 || E2counter > 0 || E3counter)
    {
        if(E1counter>0)
        {
            score += E1/E1counter;
                                printf(" E1: %f",E1/E1counter);
        }
        if(E2counter>0)
        {
            score += 0.5f * (E2/E2counter);
                                printf(" E2: %f",0.5f * (E2/E2counter));
        }
        if(E3counter>0)
        {
            score += 1.f * (E3/E3counter);
                                printf(" E3: %f",1.f * (E3/E3counter));
        }
        if(E4counter>0)
        {
            score += E4;
                                printf(" E4: %f",E4);
        }
    }
    else
    {
        score = 1E10;
    }

    printf(" Score: %f \n",score);

    return score;

}

