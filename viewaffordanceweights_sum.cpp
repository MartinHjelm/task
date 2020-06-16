// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <tuple>

// BOOST
#include "boost/program_options.hpp"

//  PCL
#include <PCLTypedefs.h>
#include <pcl/visualization/pcl_visualizer.h>

// Mine
#include <scenesegmentation.h>
#include <mainaxes.h>
#include <featurecolorquantization.h>
#include <featureopening.h>
#include <ftr_fpfh_knn.h>
#include <featurecolorhist.h>
#include <featuretexture.h>
#include <boosthelperfuns.h>
#include <eigenhelperfuns.h>



static std::unordered_map<std::string,int> affordance_map = {
	{"Brushing", 0},
	{"Containing", 1},
	{"Cutting", 2},
	{"Drinking", 3},
	{"EatingFrom", 4},
	{"Hammering", 5},
	{"HandleGrasping", 6},
	{"Hanging", 7},
	{"LiftingTop", 8},
	{"LoopGrasping", 9},
	{"Opening", 10},
	{"Playing", 11},
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


static std::map<std::string,std::tuple<int,int,int> > features_map = {
	{"ig1", std::make_tuple(1, 9, 20)},
	{"ig2", std::make_tuple(2, 20, 31)},
	{"ig3", std::make_tuple(3, 31, 42)},
	{"br",  std::make_tuple(0, 42, 53)},
	{"cq",  std::make_tuple(4, 53, 85)},
	{"f11", std::make_tuple(5, 88, 128)},
	{"f13", std::make_tuple(5, 128, 148)},
	{"f15", std::make_tuple(5, 148, 188)}
};


// Gradient functions
std::vector<int> colormap(float pos,float maxRange);
int gaussian(float x, float a, float mean, float var );


int cli_parser(int argc, char **argv, std::tuple<std::string, std::string> &affordance_args);
void printEigenVec(const Eigen::VectorXd &vec, const std::string &vecName="");


namespace po = boost::program_options;


/* MAIN */
int
main(int argc, char **argv)
{

	/************************ CLI HANDLING ************************/
	std::tuple<std::string, std::string> affordance_args;

	if (cli_parser(argc, argv, affordance_args) == 0)
		return 0;

	std::string fName = std::get<0>(affordance_args);
	std::string affordance = std::get<1>(affordance_args);

	std::cout << "Showing " << affordance << std::endl;

	std::string fName_noext = fName.substr(fName.find_last_of("/\\")+1);
	fName_noext = fName_noext.substr(0, fName_noext.length() - 4);

	std::cout << "For file: " << fName << std::endl;


	/************************ SCENE SEGMENTATION ************************/
	SceneSegmentation SS;
	SS.setInputSource(fName);
	SS.segmentPointCloud();
	SS.segmentImage();
	std::cout << "Scene segmentation done." << std::endl;



	/************************ FEATURES WEIGHTING ************************/
	int offset = 9;
	int dim = 234;
	Eigen::MatrixXd feature_weight_mat = EigenHelperFuns::readMatrixd("affordanceweights.txt");
	// Set non-used features to zero so that the normalization works.
	feature_weight_mat.col(85).setZero();
	feature_weight_mat.col(86).setZero();
	feature_weight_mat.col(87).setZero();

	Eigen::VectorXd feature_weights_vec = feature_weight_mat.row(affordance_map[affordance]).array().segment(offset, dim - offset);

    std::cout << std::endl << "Feature Weights " << feature_weights_vec.transpose() << std::endl << std::endl;
    // feature_weights_vec /= (feature_weights_vec.sum()+1E-12);
    // std::cout << "Feature Weights normalized " << feature_weights_vec.transpose() << std::endl << std::endl;
	// Remove irrelevant features.
	for(int i = 0; i < feature_weights_vec.size(); i++ )
		if(feature_weights_vec[i] < 1.E-3)
			feature_weights_vec[i] = 0;

	// Normalize
	feature_weights_vec /= (feature_weights_vec.sum()+1E-12);
	// std::cout << "Feature Weights for " << affordance << " " << featureWeightMatRow.transpose() << std::endl;
	std::cout << "Feature Weights for " << affordance << std::endl;
    printEigenVec(feature_weights_vec,"fullweights =");


	// std::cout << "Feature Weights normalized " << feature_weights_vec.transpose() << std::endl << std::endl;
	// float maxweight = 100 * feature_weights_vec.maxCoeff();



	// Make a deep non-colored copy of original point cloud
	PC::Ptr cloud_feature_itensity(new PC);
	pcl::copyPointCloud(*SS.cloudSegmented, *cloud_feature_itensity);
	for(uint iPt = 0; iPt < cloud_feature_itensity->size(); iPt++ )
	{
		cloud_feature_itensity->points[iPt].r = 0;
		cloud_feature_itensity->points[iPt].g = 0;
		cloud_feature_itensity->points[iPt].b = 0;
	}


	std::vector<float> color_weights (cloud_feature_itensity->size(), 0.);

	// For each feature
	std::map<std::string,std::tuple<int,int,int> >::const_iterator feature_it;
	for (feature_it = features_map.begin(); feature_it!=features_map.end(); ++feature_it)
	{
		// Select feature and extract weights for the feature
		std::string feature_name = feature_it->first;
		int feature_idx = std::get<0>(feature_it->second);

		int start_pos = std::get<1>(feature_it->second) - offset;
		int end_pos = std::get<2>(feature_it->second) - offset;
		int range_len = end_pos - start_pos;

		Eigen::VectorXd sub_feature_weights_vec = feature_weights_vec.array().segment(start_pos, range_len );
		std::cout << "Doing feature " << feature_name << "("<< feature_idx << ")" << " in range "
		<< start_pos << "-" << end_pos << " with length " << range_len << std::endl;

		printEigenVec(sub_feature_weights_vec, feature_name + " =");

		// sub_feature_weights_vec /= (sub_feature_weights_vec.sum()+1E-12);



		/*** COMPUTE SELECTED FEATURE ***/

		FeatureColorHist FCH;
		if(feature_name.compare("ig1") == 0 || feature_name.compare("ig2") == 0 || feature_name.compare("ig3") == 0 || feature_name.compare("br") == 0 )
		{
			FCH.setInputSource(SS.imgSegmented);
			FCH.computeFeatureMats();
		}

		FeatureColorQuantization FCQ;
		if(feature_name.compare("cq") == 0)
		{
			FCQ.setInputSource(SS.rawfileName,SS.imgROI,SS.roiObjImgIndices,SS.offset_,SS.cloudSegmented);
			FCQ.colorQuantize();
			FCQ.imgCQ2PC();
		}

		// Best combination 0.02_0.02_20_20 - 0.02_0.06_40_0 - 0.01_0.05_40_20
		FeatureFPFHBoW Ffpfh;
		Ffpfh.SetInputSource(SS.cloudSegmented,SS.cloudSegmentedNormals);
		if(feature_name.compare("f11") == 0)
		{
			Ffpfh.SetCodeBook("0.01", "0.05", "40", "20");
			Ffpfh.CptBoWRepresentation();
		}
		else if(feature_name.compare("f13") == 0)
		{
			Ffpfh.SetCodeBook("0.02", "0.02", "20", "20");
			Ffpfh.CptBoWRepresentation();
		}
		else if(feature_name.compare("f15") == 0)
		{
			Ffpfh.SetCodeBook("0.02", "0.06", "40", "0");
			Ffpfh.CptBoWRepresentation();
		}




		/*** COMPUTE POINT CLOUD POINT INTESITY ***/

		// Eigen::VectorXd itensHist = Eigen::VectorXd::Zero(255);
		// cv::Mat HSL(1,1, CV_8UC3, cv::Scalar(0,0,255));
		// cv::Mat RGB(1,1, CV_8UC3, cv::Scalar(0,0,255));

		int pt_feature_idx = 0;
		for(uint iPt = 0; iPt < cloud_feature_itensity->size(); iPt++ )
		{
			// std::cout << SS.cloudSegmented->points[iPt].ID << " " << feature_idx << std::endl;
			switch(feature_idx)
			{
			case 0: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID, 0, 640); break;
			case 1: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID, 1, 640); break;
			case 2: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID, 2, 640); break;
			case 3: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID, 3, 640); break;
			case 4: pt_feature_idx = FCQ.pcCQIdxs[iPt]; break;
			case 5: pt_feature_idx = Ffpfh.GetBoWForPoint(iPt); break;
			}

			// std::cout << pt_feature_idx << ", " << std::endl;
			// Copy RGB values from pt
			// RGB.at<cv::Vec3b>(0,0)[0] = 255.;//SS.cloudSegmented->points[iPt].r;
			// RGB.at<cv::Vec3b>(0,0)[1] = 255.;//SS.cloudSegmented->points[iPt].g;
			// RGB.at<cv::Vec3b>(0,0)[2] = 255.;//SS.cloudSegmented->points[iPt].b;
			// // Cnvert RGB to HSL
			// cvtColor(RGB, HSL, CV_RGB2HLS);
			// // Adjust brightness
			// HSL.at<cv::Vec3b>(0,0)[1] *= sub_feature_weights_vec[pt_feature_idx];
			//
			// // Compute intensity histogram
			// for(uint i = 0; i < (uint)HSL.at<cv::Vec3b>(0,0)[1]; i++)
			//         itensHist[i]++;
			//
			// // Cnvert back
			// cvtColor(HSL, RGB,CV_HLS2RGB);

			// cloud_feature_itensity->points[iPt].r = (int) RGB.at<cv::Vec3b>(0,0)[0];
			// cloud_feature_itensity->points[iPt].g = (int) RGB.at<cv::Vec3b>(0,0)[1];
			// cloud_feature_itensity->points[iPt].b = (int) RGB.at<cv::Vec3b>(0,0)[2];

				color_weights[iPt] += sub_feature_weights_vec[pt_feature_idx];
			// std::cout << 100.*feature_weights_vec[pt_feature_idx] << " " << rgb[0] << " " << rgb[1] << " " << rgb[2] << std::endl;

		}

	}

	float max_val = *std::max_element(color_weights.begin(), color_weights.end());
	std::cout << max_val << std::endl;
	std::transform(color_weights.begin(), color_weights.end(), color_weights.begin(),
               std::bind1st(std::multiplies<float>(), 1./max_val));

	int pt_feature_idx = 0;
	for(uint iPt = 0; iPt < cloud_feature_itensity->size(); iPt++ )
	{
		// std::cout << color_weights[iPt] << std::endl;
		std::vector<int> rgb = colormap( 100. * color_weights[iPt], 100.);
		// std::cout << color_weights[iPt] <<  ": " << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << std::endl;
		cloud_feature_itensity->points[iPt].r += rgb[0];
		cloud_feature_itensity->points[iPt].g += rgb[1];
		cloud_feature_itensity->points[iPt].b += rgb[2];
	}






	/************************ VISUALIZATION ************************/
	//Add 3 view ports
	int v1(0), v2(0), v3(0);
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (1.0, 1.0, 1.0);
	viewer->initCameraParameters ();
	viewer->setCameraPosition(0,0,0,0,-1,1,0);

	// Viewport 1
	viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(SS.cloudScene);
	viewer->addPointCloud<PointT> (SS.cloudScene, rgb1, "Object",v1);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10,"Object");
	viewer->setBackgroundColor (1., 1., 1.,v1);
	//    viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

	// Viewport 3
	viewer->createViewPort (0.5, 0.0, 1., 1.0, v3);
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(cloud_feature_itensity);
	viewer->addPointCloud<PointT> (cloud_feature_itensity, rgb3, "Object Feature Intensity",v3);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10,"Object Feature Intensity");
	viewer->setBackgroundColor (1., 1., 1.,v3);

	viewer->resetCamera();

	// viewer->setCameraPosition(-0.0555625, -0.136915, 0.035341,-0.0729603, 0.087453, 0.609634,-0.00663999, -0.931486, 0.363716, v3);
	// std::string fname = ;
	viewer->spinOnce (1);

	boost::filesystem::path dir("feature_weight_plots/"+affordance);
	boost::filesystem::create_directories(dir);

	// fName.find_last_of(".png")

	viewer->saveScreenshot("feature_weight_plots/"+affordance+"/"+fName_noext+"_paper.png");
	// viewer->close();

	viewer->setWindowName(affordance + " - " + fName_noext);

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (10000));
	}

	return 1;

}




/************************ HELPER FUNCTIONS ************************/

int gaussian(float x, float a, float mean, float var )
{
	return (int) a * exp(-pow(x - mean,2.) / (2. * pow(var,2.)));
}


std::vector<int> colormap(float pos, float maxRange)
{
	std::vector<int> rgb = {0,0,0};
	rgb[0] = gaussian(pos, 255., maxRange, maxRange * 0.5);
	rgb[1] = 0;
	rgb[2] = gaussian(pos, 255., 0., maxRange * 0.5);
	return rgb;
}



void printEigenVec(const Eigen::VectorXd &vec, const std::string &vecName)
{
  using namespace std;
  if(vecName.size()!=0)
    cout << vecName << " [";
      else
        cout << "[";
          for(int idx=0; idx!=vec.size()-1; idx++) { cout << vec(idx) << ", "; }
              cout << vec(vec.size()-1) << "]" << endl << endl;
}




int cli_parser(int argc, char **argv, std::tuple<std::string, std::string> &affordance_args)
{

	// std::tuple<std::string, std::string> affordance_args{'',''};

	// Arguments set up
	po::options_description desc("Allowed options");
	desc.add_options()
	  ("help","Produces the function descriptions")
	  ("fn",po::value<std::string>(), "filename of file ")
	  ("a",po::value<std::string>(), "the affordance to view")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help") || argc == 1) {
		std::cout << desc << "\n";
		std::cout << "Affordances:" << std::endl;
		for ( auto it = affordance_map.begin(); it != affordance_map.end(); ++it )
			std::cout << " " << it->first << ": " << it->second <<  std::endl;
		return 0;
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
		std::unordered_map<std::string,int>::const_iterator gotAffordance = affordance_map.find (affordance);
		if ( gotAffordance == affordance_map.end() )
		{
			std::cout << "Affordance " << affordance << " not found exiting..." << std::endl;
			return 0;
		}

	}
	else
	{
		std::cout << "Specify affordance. Please!" << std::endl;
		std::cout << "Affordances:" << std::endl;
		for ( auto it = affordance_map.begin(); it != affordance_map.end(); ++it )
			std::cout << " " << it->first << ": " << it->second <<  std::endl;

		return 0;
	}


	std::get<0>(affordance_args) = fName;
	std::get<1>(affordance_args) = affordance;
	return 1;
}
