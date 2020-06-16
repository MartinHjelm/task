// std
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <tuple>

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
	{"f13", std::make_tuple(6, 128, 148)},
	{"f15", std::make_tuple(7, 148, 188)}
};


static const int offset = 9;
static const int dimension = 234;



/* COLOR MAPPING FUNCTIONS  */
std::vector<int> colormap(float pos,float maxRange);
int gaussian(float x, float a, float mean, float var );

int gaussian(float x, float a, float mean, float var )
{
	return (int) a * exp(-pow(x - mean,2) / (2. * pow(var,2)));
}


std::vector<int> colormap(float pos, float maxRange)
{
	std::vector<int> rgb = {0,0,0};
	rgb[0] = gaussian(pos, 255., maxRange, maxRange * 0.5);
	rgb[1] = 0;
	rgb[2] = gaussian(pos, 255., 0., maxRange * 0.5 );
	return rgb;
}




/* MAIN */
int
main(int argc, char **argv)
{


	Eigen::MatrixXi label_mat = EigenHelperFuns::readCSV2Matrix("labels.csv").cast <int> ();
	Eigen::MatrixXd feature_weight_mat = EigenHelperFuns::readMatrixd("affordanceweights.txt");

	// Set these to zero so normalization will work later on.
	feature_weight_mat.col(85).setZero();
	feature_weight_mat.col(86).setZero();
	feature_weight_mat.col(87).setZero();


	/** Set path names. **/
	/* The scence directory contains the
	 * pcd files containing recordings of scenes with just
	 * one object present. The scene_grasp directory contains
	 * files with the name foo1.pcd, foo2.pcd and so on. Where
	 * foo referes to a scene file in the scene directory.
	 */


	// 1. Find all files in non-action scene recordings directory
	std::string dirname = argv[1];
	boost::filesystem::path scencesDirName (dirname);
	std::vector<boost::filesystem::path> scene_fnames;
	std::vector<std::string> exts = {".pcd"};
	BoostHelperFuns::getListOfFilesInDir(scencesDirName, exts, scene_fnames);

	int file_counter = 0;

	/** FOR EACH OBJECT... **/
	std::vector<boost::filesystem::path>::iterator fname_ptr;
	for(fname_ptr = scene_fnames.begin(); fname_ptr!=scene_fnames.end(); ++fname_ptr)
	{
		// if (file_counter < 200 )
		// {
		// 	file_counter++;
		// 	continue;
		// }

		std::cout << "File counter " << fname_ptr->string() << std::endl;
		std::string fname = fname_ptr->string();
		std::string sceneFileFullPath = scencesDirName.string() + fname_ptr->string();
		std::cout << sceneFileFullPath << std::endl;



		/************************ SCENE SEGMENTATION ************************/
		SceneSegmentation SS;
		SS.setInputSource(sceneFileFullPath);
		SS.segmentPointCloud();
		SS.segmentImage();
		std::cout << "Scene segmentation done." << std::endl;



		/************************ COMPUTE FEATURES ************************/
		FeatureColorHist FCH;
		FCH.setInputSource(SS.imgSegmented);
		FCH.computeFeatureMats();

		FeatureColorQuantization FCQ;
		FCQ.setInputSource(SS.rawfileName,SS.imgROI,SS.roiObjImgIndices,SS.offset_,SS.cloudSegmented);
		FCQ.colorQuantize();
		FCQ.imgCQ2PC();

		FeatureFPFHBoW Ffpfh11;
		Ffpfh11.SetInputSource(SS.cloudSegmented,SS.cloudSegmentedNormals);
		Ffpfh11.SetCodeBook("0.01", "0.05", "40", "20");
		Ffpfh11.CptBoWRepresentation();

		FeatureFPFHBoW Ffpfh13;
		Ffpfh13.SetInputSource(SS.cloudSegmented,SS.cloudSegmentedNormals);
		Ffpfh13.SetCodeBook("0.02", "0.02", "20", "20");
		Ffpfh13.CptBoWRepresentation();

		FeatureFPFHBoW Ffpfh15;
		Ffpfh15.SetInputSource(SS.cloudSegmented,SS.cloudSegmentedNormals);
		Ffpfh15.SetCodeBook("0.02", "0.06", "40", "0");
		Ffpfh15.CptBoWRepresentation();




		// FOR EACH AFFORDANCE
		for ( auto affordance_ptr = affordance_map.begin(); affordance_ptr != affordance_map.end(); ++affordance_ptr )
		{
			// Only do coloring for positive labeled examples
			if(label_mat(file_counter,affordance_ptr->second)==0)
				continue;

			std::string affordance = affordance_ptr->first;
			std::cout << std::endl << "AFFORDANCE: " << affordance << std::endl;


			// Make a deep copy of original point cloud and reset colors
			PC::Ptr cloud_feature_itensity(new PC);

			pcl::copyPointCloud(*SS.cloudSegmented, *cloud_feature_itensity);
			for(uint iPt = 0; iPt < cloud_feature_itensity->size(); iPt++ )
			{
				cloud_feature_itensity->points[iPt].r = 0;
				cloud_feature_itensity->points[iPt].g = 0;
				cloud_feature_itensity->points[iPt].b = 0;
			}


			// GET AFFORDANCE FEATURE WEIGHTS
			Eigen::VectorXd feature_weights_vec = feature_weight_mat.row(affordance_map[affordance]).array().segment(offset, dimension-offset);

			// // Normalize
			// feature_weights_vec /= (feature_weights_vec.sum()+1E-12);

			// // Remove irrelevant features.
			// for(int i = 0; i < feature_weights_vec.size(); i++ )
			// 	if(feature_weights_vec[i]<0.01)
			// 		feature_weights_vec[i] = 0;

			// // Normalize again
			// feature_weights_vec /= (feature_weights_vec.sum()+1E-12);
			// std::cout << "Feature Weights for " << affordance << " " << featureWeightMatRow.transpose() << std::endl;
			// std::cout << "Feature Weights normalized  " << feature_weights_vec << std::endl;


			// float max_weight = 100 * feature_weights_vec.maxCoeff();
			std::vector<float> color_weights (cloud_feature_itensity->size(), 0.);

			std::map<std::string,std::tuple<int, int, int> >::const_iterator feature_it;
			for(feature_it = features_map.begin(); feature_it!=features_map.end(); ++feature_it)
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

				if(sub_feature_weights_vec.sum()<=0.)
					continue;

				int pt_feature_idx = 0;
				for(uint iPt = 0; iPt < cloud_feature_itensity->size(); iPt++ )
				{
					// std::cout << SS.cloudSegmented->points[iPt].ID << " " << feature_idx << std::endl;
					switch(feature_idx)
					{
					case 0: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID,0,640); break;
					case 1: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID,1,640); break;
					case 2: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID,2,640); break;
					case 3: pt_feature_idx = FCH.binAtPoint(cloud_feature_itensity->points[iPt].ID,3,640); break;
					case 4: pt_feature_idx = FCQ.pcCQIdxs[iPt]; break;
					case 5: pt_feature_idx = Ffpfh11.GetBoWForPoint(iPt); break;
					case 6: pt_feature_idx = Ffpfh13.GetBoWForPoint(iPt); break;
					case 7: pt_feature_idx = Ffpfh15.GetBoWForPoint(iPt); break;
					}

					color_weights[iPt] += sub_feature_weights_vec[pt_feature_idx];

				}
				std::cout << std::endl;
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

			/************************ ADD VIEWPORTS ************************/
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
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 11,"Object");
			viewer->setBackgroundColor (1., 1., 1.,v1);
			//    viewer->addPointCloudNormals<PointT, pcl::Normal>(SS.cloudSegmented, SS.cloudSegmentedNormals, 10, 0.01, "Normals", v1);

			// Viewport 3
			viewer->createViewPort (0.5, 0.0, 1., 1.0, v3);
			pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(cloud_feature_itensity);
			viewer->addPointCloud<PointT> (cloud_feature_itensity, rgb3, "Object Feature Intensity",v3);
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 11,"Object Feature Intensity");
			viewer->setBackgroundColor (1., 1., 1.,v3);

			viewer->resetCamera();

			// viewer->setCameraPosition(-0.0555625, -0.136915, 0.035341,-0.0729603, 0.087453, 0.609634,-0.00663999, -0.931486, 0.363716, v3);
			// std::string fname = ;

			viewer->spinOnce (1);

			boost::filesystem::path dir("feature_weight_plots/"+affordance);
			boost::filesystem::create_directories(dir);

			viewer->saveScreenshot("feature_weight_plots/"+affordance+"/"+fname.substr(0, fname.length()-4)+".png");
			viewer->close();

		}

		file_counter++;
	}


// while (!viewer->wasStopped ())
// {
//         viewer->spinOnce (100);
//         boost::this_thread::sleep (boost::posix_time::microseconds (10000));
// }

	return 1;

}
