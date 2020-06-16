// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <unordered_map>
#include <math.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// BOOST
#include "boost/program_options.hpp"

//  PCL
#include <PCLTypedefs.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>

// Mine
#include <scenesegmentation.h>
#include <mainaxes.h>
#include <featurecolorquantization.h>
#include <featureopening.h>
#include <featurefpfh.h>
#include <featurecolorhist.h>
#include <featuretexture.h>
#include <myhelperfuns.h>
#include <boosthelperfuns.h>
#include <pclhelperfuns.h>
#include <eigenhelperfuns.h>
#include <symmetrycompletion.h>



static std::unordered_map<std::string,int> affordanceMap = {
        {"Brushing",0},
        {"Containing",1},
        {"Cutting",2},
        {"Drinking",3},
        {"EatingFrom",4},
        {"Hammering",5},
        {"HandleGrasping",6},
        {"Hanging",7},
        {"LiftingTop",8},
        {"LoopGrasping",9},
        {"Opening",10},
        {"Playing",11},
        {"Pounding",12},
        {"Pouring",13},
        {"Putting",14},
        {"Rolling",15},
        {"Scraping",16},
        {"Shaking",17},
        {"Spraying",18},
        {"Squeezing",19},
        {"SqueezingOut",20},
        {"Stacking",21},
        {"Stirring",22},
        {"Tool",23},
        {"Writing",24}
};


static std::unordered_map<std::string,std::pair<int,int> > featureRangesMap = {
        {"ig1", std::make_pair(10,21) },
        {"ig2", std::make_pair(21, 32)},
        {"ig3", std::make_pair(32, 43)},
        {"br", std::make_pair(43, 54)},
        {"cq", std::make_pair(54, 70)},
        {"f11", std::make_pair(73, 113)},
        {"f13", std::make_pair(113, 133)},
        {"f15", std::make_pair(133, 173)},
        {"hog", std::make_pair(173, 213)}
};

namespace po = boost::program_options;


std::vector<int> colormap(float pos ,float maxRange);
int gaussian(float x, float a, float mean, float var );

int gaussian(float x, float a, float mean, float var )
{
        return (int) a * exp(- pow(x - mean,2) / (2 * pow(var,2)));
}


std::vector<int> colormap(float pos, float maxRange)
{
        std::vector<int> rgb = {0,0,0};
        rgb[0] = gaussian(pos, 255., maxRange, maxRange * 0.35);
        rgb[1] = 0; 
        rgb[2] = gaussian(pos, 255., 0., maxRange * 0.35);
        return rgb;
}


/* MAIN */
int
main(int argc, char **argv)
{
        // Arguments set up
        po::options_description desc("Allowed options");
        desc.add_options()
                ("help","Produces the function descriptions")
                ("fn",po::value<std::string>(), "filename of file ")
                ("a",po::value<std::string>(), "the affordance to view")
                ("f",po::value<std::string>(), "the feature to view")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
                std::cout << desc << "\n";
                std::cout << "Affordances:" << std::endl;
                for ( auto it = affordanceMap.begin(); it != affordanceMap.end(); ++it )
                        std::cout << " " << it->first << ": " << it->second <<  std::endl;
                std::cout << "Features: " << std::endl;
                for ( auto it = featureRangesMap.begin(); it != featureRangesMap.end(); ++it )
                        std::cout << " " << it->first << ": (" << it->second.first << "," << it->second.second << ")" << std::endl;
                return 1;
        }

        std::string fName;
        if(vm.count("fn"))
        {
                fName = vm["fn"].as<std::string>();
                if(!BoostHelperFuns::fileExist(fName))
                {
                        std::cout << "Point cloud file not found!" << std::endl;
                        return 0;
                }

        }
        else
        {
                std::cout << "Specify file name. Please!" << std::endl;
                return 0;
        }

        std::string affordance;
        if(vm.count("a"))
        {
                affordance = vm["a"].as<std::string>();
                std::unordered_map<std::string,int>::const_iterator gotAffordance = affordanceMap.find (affordance);
                if ( gotAffordance == affordanceMap.end() )
                {
                        std::cout << "Affordance " << affordance << " not found exiting..." << std::endl;
                        return 0;
                }

        }
        else
        {
                std::cout << "Specify affordance. Please!" << std::endl;
                std::cout << "Affordances:" << std::endl;
                for ( auto it = affordanceMap.begin(); it != affordanceMap.end(); ++it )
                        std::cout << " " << it->first << ": " << it->second <<  std::endl;

                return 0;
        }


        std::string feature;
        if(vm.count("f"))
        {
                feature = vm["f"].as<std::string>();
                std::unordered_map<std::string,std::pair<int,int> >::const_iterator gotFeature = featureRangesMap.find (feature);
                if ( gotFeature == featureRangesMap.end() )
                {
                        std::cout << "Feature " << feature << " not found exiting..." << std::endl;
                        return 0;
                }
        }
        else
        {
                std::cout << "Specify feature. Please!" << std::endl;
                std::cout << "Features: " << std::endl;
                for ( auto it = featureRangesMap.begin(); it != featureRangesMap.end(); ++it )
                        std::cout << " " << it->first << ": (" << it->second.first << "," << it->second.second << ")" << std::endl;

                return 0;
        }

        std::cout << "Showing " << affordance << " with feature " << feature << std::endl;

        std::clock_t startTime; // Timer

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


        /************************ WEIGHT FEATURES ************************/

        Eigen::MatrixXd featureWeightMatrix = EigenHelperFuns::readMatrixd("affordanceweights.txt");
        Eigen::VectorXd featureWeightMatRow = featureWeightMatrix.row(affordanceMap[affordance]);
        Eigen::VectorXd featureWeights = featureWeightMatRow.array().segment(featureRangesMap[feature].first,featureRangesMap[feature].second-featureRangesMap[feature].first);

        std::cout << "Feature Weights for " << affordance << " " << featureWeightMatRow.transpose() << std::endl;
        std::cout << "Feature Weights for " << feature << " " << featureWeights.transpose() << std::endl;

        // std::cout << "Feature Weights Orig"<< " (rows,cols), (" << featureWeightMatRow.rows()
        //             << ","<< featureWeightMatRow.cols() << "), " << ":" <<
        //                featureWeightMatRow.transpose() << std::endl;

        featureWeights /= (featureWeights.sum()+1E-12);

        for(int idx = 0; idx!=featureWeights.size(); idx++)
                if(featureWeights[idx]<5.E-2)
                        featureWeights[idx] = 0.;   

        featureWeights /= (featureWeights.sum()+1E-12);
        float maxweight = 100 * featureWeights.maxCoeff();


        std::cout << "Feature Weights Normalized "<< featureWeights.transpose() << std::endl;
        std::cout << "Feature Weights Sum "<< featureWeights.sum() << std::endl;

        int featureIdx = 0;


        // FILTER
        FeatureColorHist FCH;
        if(feature.compare("ig1") == 0 || feature.compare("ig2") == 0 || feature.compare("ig3") == 0 || feature.compare("br") == 0 )
        {
                if(feature.compare("br") == 0)
                        featureIdx = 0;
                else if(feature.compare("ig1") == 0)
                        featureIdx = 1;
                else if(feature.compare("ig2") == 0)
                        featureIdx = 2;
                else if(feature.compare("ig3") == 0)
                        featureIdx = 3;

                FCH.setInputSource(SS.imgSegmented);
                FCH.computeFeatureMats();
        }
        // COLOR
        FeatureColorQuantization FCQ;
        if(feature.compare("cq") == 0)
        {
                featureIdx = 4;
                FCQ.setInputSource(SS.rawfileName,SS.imgROI,SS.roiObjImgIndices,SS.offset_,SS.cloudSegmented);
                FCQ.colorQuantize();
                FCQ.imgCQ2PC();
        }
        //V3f cqVals = FCQ.pcCQIdxs[iter];


        // Best combination 0.02_0.02_20_20 - 0.02_0.06_40_0 - 0.01_0.05_40_20
        // BOW
        featureFPFH Ffpfh;
        Ffpfh.setInputSource(SS.cloudSegmented,SS.cloudSegmentedNormals);
        if(feature.compare("f11") == 0)
        {
                featureIdx = 5;
                Ffpfh.setCodeBook("0.01", "0.05", "40", "20");
                Ffpfh.cptBoWRepresentation();
        }
        else if(feature.compare("f13") == 0)
        {
                featureIdx = 5;
                Ffpfh.setCodeBook("0.02", "0.02", "20", "20");
                Ffpfh.cptBoWRepresentation();
        }
        else if(feature.compare("f15") == 0)
        {
                featureIdx = 5;
                Ffpfh.setCodeBook("0.02", "0.06", "40", "0");
                Ffpfh.cptBoWRepresentation();
        }







        /* COMPUTE THE INTESITY MAPPING */
        PC::Ptr cloudFeatureItensity(new PC);
        // Make a deep copy of original point cloud
        pcl::copyPointCloud(*SS.cloudSegmented, *cloudFeatureItensity);

        Eigen::VectorXd itensHist = Eigen::VectorXd::Zero(255);

        cv::Mat HSL(1,1, CV_8UC3, cv::Scalar(0,0,255));
        cv::Mat RGB(1,1, CV_8UC3, cv::Scalar(0,0,255));

        int ptFeatIdx = 0;
        for(uint iPt = 0; iPt < cloudFeatureItensity->size(); iPt++ )
        {
                // std::cout << SS.cloudSegmented->points[iPt].ID << " " << featureIdx << std::endl;
                switch(featureIdx)
                {
                case 0: ptFeatIdx = FCH.binAtPoint(cloudFeatureItensity->points[iPt].ID,0,640); break;
                case 1: ptFeatIdx = FCH.binAtPoint(cloudFeatureItensity->points[iPt].ID,1,640); break;
                case 2: ptFeatIdx = FCH.binAtPoint(cloudFeatureItensity->points[iPt].ID,2,640); break;
                case 3: ptFeatIdx = FCH.binAtPoint(cloudFeatureItensity->points[iPt].ID,3,640); break;
                case 4: ptFeatIdx = FCQ.pcCQIdxs[iPt]; break;
                case 5: ptFeatIdx = Ffpfh.getBoWForPoint(iPt); break;
                }

                // std::cout << ptFeatIdx << ", " << std::endl;
                // Copy RGB values from pt
                RGB.at<cv::Vec3b>(0,0)[0] = 255.;//SS.cloudSegmented->points[iPt].r;
                RGB.at<cv::Vec3b>(0,0)[1] = 255.;//SS.cloudSegmented->points[iPt].g;
                RGB.at<cv::Vec3b>(0,0)[2] = 255.;//SS.cloudSegmented->points[iPt].b;
                // Cnvert RGB to HSL
                cvtColor(RGB, HSL, CV_RGB2HLS);
                // Adjust brightness
                HSL.at<cv::Vec3b>(0,0)[1] *= featureWeights[ptFeatIdx];

                // Compute intensity histogram
                for(uint i = 0; i < (uint)HSL.at<cv::Vec3b>(0,0)[1]; i++)
                        itensHist[i]++;

                // Cnvert back
                cvtColor(HSL, RGB,CV_HLS2RGB);

                cloudFeatureItensity->points[iPt].r = (int) RGB.at<cv::Vec3b>(0,0)[0];
                cloudFeatureItensity->points[iPt].g = (int) RGB.at<cv::Vec3b>(0,0)[1];
                cloudFeatureItensity->points[iPt].b = (int) RGB.at<cv::Vec3b>(0,0)[2];
                
                std::vector<int> rgb = colormap( 100.*featureWeights[ptFeatIdx],maxweight);

                cloudFeatureItensity->points[iPt].r = rgb[0];
                cloudFeatureItensity->points[iPt].g = rgb[1];
                cloudFeatureItensity->points[iPt].b = rgb[2];   
                // std::cout << 100.*featureWeights[ptFeatIdx] << " " << rgb[0] << " " << rgb[1] << " " << rgb[2] << std::endl;                

        }

        // // // Normalize
        // itensHist /= SS.cloudSegmented->size();

        // std::cout << "Itensity histogram " << itensHist.transpose() << std::endl;

        // for(uint iPt = 0; iPt < SS.cloudSegmented->size(); iPt++ )
        // {
        //         RGB.at<cv::Vec3b>(0,0)[0] = cloudFeatureItensity->points[iPt].r;
        //         RGB.at<cv::Vec3b>(0,0)[1] = cloudFeatureItensity->points[iPt].g;
        //         RGB.at<cv::Vec3b>(0,0)[2] = cloudFeatureItensity->points[iPt].b;
        //         // Cnvert RGB to HSL
        //         cvtColor(RGB, HSL, CV_RGB2HLS);
        //         // Adjust brightness
        //         HSL.at<cv::Vec3b>(0,0)[1] = 255. * itensHist[ HSL.at<cv::Vec3b>(0,0)[1] ];
        //         cvtColor(HSL, RGB,CV_HLS2RGB);

        //         std::vector<float> rgb = colormap( 255.*itensHist[ HSL.at<cv::Vec3b>(0,0)[1] ]);
        //         cloudFeatureItensity->points[iPt].r = RGB.at<cv::Vec3b>(0,0)[0];
        //         cloudFeatureItensity->points[iPt].g = RGB.at<cv::Vec3b>(0,0)[1];
        //         cloudFeatureItensity->points[iPt].b = RGB.at<cv::Vec3b>(0,0)[2];

        //         // cloudFeatureItensity->points[iPt].r = rgb[0];
        //         // cloudFeatureItensity->points[iPt].g = rgb[1];
        //         // cloudFeatureItensity->points[iPt].b = rgb[2];

        //         std::cout << 255.*itensHist[ HSL.at<cv::Vec3b>(0,0)[1] ] << " " << rgb[0] << " " << rgb[1] << " " << rgb[2] << std::endl;                
        // }


				
        /* COMPUTE THE FEATURE MAPPING */
        PC::Ptr cloudFeatureMapping(new PC);
        // Make a deep copy of original point cloud
        pcl::copyPointCloud(*SS.cloudSegmented, *cloudFeatureMapping);

        if(feature.compare("cq")==0)
        {
                //Colorize object according to quantization
                for(uint iter = 0; iter!=cloudFeatureMapping->size(); iter++)
                {
                        V3f cqVals = FCQ.pcCQVals[iter];
                        //            EigenHelperFuns::printEigenVec(cqVals,"Color ");
                        cloudFeatureMapping->points[iter].r = cqVals(0);
                        cloudFeatureMapping->points[iter].g = cqVals(1);
                        cloudFeatureMapping->points[iter].b = cqVals(2);
                }
        }


        if(feature.compare("ig1") == 0 || feature.compare("ig2") == 0 || feature.compare("ig3") == 0 || feature.compare("br") == 0 )
        {
                int fIdx = 0;
                if(feature.compare("br") == 0)
                        fIdx = 0;
                else if(feature.compare("ig1") == 0)
                        fIdx = 1;
                else if(feature.compare("ig2") == 0)
                        fIdx = 2;
                else if(feature.compare("ig3") == 0)
                        fIdx = 3;

                for(size_t iVal=0; iVal!=cloudFeatureMapping->points.size(); ++iVal)
                {
                        // std::cout << "Point ID " << iVal << " at "  << SS.cloudSegmented->points[iVal].ID << '\n';
                        double grad = FCH.gradAtPoint(cloudFeatureMapping->points[iVal].ID,fIdx,640);
                        // std::cout << grad << std::endl;
                        cloudFeatureMapping->points[iVal].r = 255*(grad/255.0);
                        cloudFeatureMapping->points[iVal].g = 0.;
                        cloudFeatureMapping->points[iVal].b = 0.;
                }
        }



        if(feature.compare("f11") == 0 || feature.compare("f13") == 0 || feature.compare("f15") == 0)
        {
                // Color object according to BoW code.
                for(uint iVal=0; iVal!=cloudFeatureMapping->size(); ++iVal)
                {
                        std::vector<int> rgb =  MyHelperFuns::getColor(Ffpfh.getBoWForPoint(iVal));
                        cloudFeatureMapping->points[iVal].r = rgb[0];
                        cloudFeatureMapping->points[iVal].g = rgb[1];
                        cloudFeatureMapping->points[iVal].b = rgb[2];
                }
        }





        /************************ ADD VIEWPORTS ************************/
        //Add 3 view ports
        int v1(0), v2(0), v3(0);
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->setBackgroundColor (1.0, 1.0, 1.0);
        viewer->initCameraParameters ();
        viewer->setCameraPosition(0,0,0,0,-1,1,0);

        // Viewport 1
        viewer->createViewPort (0.0, 0.0, 0.33, 1.0, v1);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(SS.cloudScene);
        viewer->addPointCloud<PointT> (SS.cloudScene, rgb1, "Object",v1);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Object");
        //    viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

        // Viewport 2
        viewer->createViewPort (0.33, 0.0, .66, 1.0, v2);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(cloudFeatureMapping);
        viewer->addPointCloud<PointT> (cloudFeatureMapping, rgb2, "Object Feature",v2);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Object Feature");
        // viewer->setBackgroundColor (.5, 0., 0.,v2);

        // Viewport 3
        viewer->createViewPort (0.66, 0.0, 1., 1.0, v3);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(cloudFeatureItensity);
        viewer->addPointCloud<PointT> (cloudFeatureItensity, rgb3, "Object Feature Intensity",v3);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"Object Feature Intensity");
        // viewer->setBackgroundColor (.5, 0., 0.,v3);

        while (!viewer->wasStopped ())
        {
                viewer->spinOnce (100);
                boost::this_thread::sleep (boost::posix_time::microseconds (10000));
        }

        return 1;

}
