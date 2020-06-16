#include "ftr_fpfh_knn.h"

// STD
#include <stdexcept>
#include <sstream>
#include <stdlib.h>     /* atoi */

// PCL
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/impl/fpfh_omp.hpp>

// Mine
#include <myhelperfuns.h>
#include <boosthelperfuns.h>
#include <eigenhelperfuns.h>
#include <pclhelperfuns.h>
// BoW code
#include <kmeans.h>
#include <bagofwords.h>


FeatureFPFHBoW::FeatureFPFHBoW() {
}


void
FeatureFPFHBoW::SetInputSource(const PC::Ptr &cPtr,const PCN::Ptr &cnPtr)
{
	cloud_ = cPtr;
	cloudNormals_ = cnPtr;
}

void
FeatureFPFHBoW::SetCodeBook( const std::string &normalRadius, const std::string &fpfhRadius, const std::string &codeBookSize, const std::string &fpfhProjDim )
{

	normalRadiusSearch_ = std::stof(normalRadius);
	fpfhRadiusSearch_ = std::stof(fpfhRadius);
	codeBookSize_ = std::stoi(codeBookSize);
	fpfhProjDim_ = std::stoi(fpfhProjDim);

	std::string fileParamString = normalRadius+"_"+fpfhRadius+"_"+codeBookSize+"_"+fpfhProjDim;
	std::string fnCodeBook = std::string("bowmodels/knn_codebook_")+fileParamString+".txt";
	std::string fnPCAProjMat = "bowmodels/knn_projmat_"+fileParamString+".txt";

	// Get codebook and projection matrix file names
	if(!BoostHelperFuns::fileExist(fnCodeBook))
		throw std::runtime_error("Could not find "+fnCodeBook);
	if(fpfhProjDim_ > 0 && !BoostHelperFuns::fileExist(fnPCAProjMat))
		throw std::runtime_error("Could not find "+fnPCAProjMat);

	codeBook_ = EigenHelperFuns::readMatrixd(fnCodeBook);
	if(fpfhProjDim_ > 0)
		pcaProjMat_ = EigenHelperFuns::readMatrixd(fnPCAProjMat);
}


// Computes BoW representation for each point on object
void
FeatureFPFHBoW::CptBoWRepresentation()
{
	ComputeFPFHFeatures();
	EncodeFeatures();
}


/* Computes the FPFH feature over all points in the point cloud and
 * projects the features using PCA. */
void
FeatureFPFHBoW::ComputeFPFHFeatures()
{
	// Recompute cloud normals since we require a special setting for fpfh
	// PCN::Ptr cloudNormals(new PCN);
	// PCLHelperFuns::computeCloudNormals(cloud_,normalRadiusSearch_,cloudNormals);

	// Do FPFH estimation
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
	pcl::FPFHEstimationOMP<PointT, PointN, pcl::FPFHSignature33> fpfh;
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	fpfh.setInputCloud (cloud_);
	fpfh.setInputNormals (cloudNormals_);
	fpfh.setSearchMethod (tree);
	fpfh.setRadiusSearch (fpfhRadiusSearch_);
	fpfh.compute (*fpfhs);

	// Create an Eigen mat to store this objects fpfh's
	featureMat_.resize(fpfhs->points.size(),33);
	for(uint iIdx = 0; iIdx < fpfhs->points.size(); iIdx++)
		for(uint iVal = 0; iVal < 33; iVal++)
			featureMat_(iIdx,iVal) = fpfhs->points[iIdx].histogram[iVal];

	// Do pca projection of the features
	//    EigenHelperFuns::printMatSized(featureMat_,"Featuremat");
	//    EigenHelperFuns::printMatSized(pcaProjMat_,"PCA mat");
	if(fpfhProjDim_ > 0)
		featureMat_ = featureMat_ * pcaProjMat_;
}


// Encodes a feature vector to BoW indices
void
FeatureFPFHBoW::EncodeFeatures()
{
	// Create and init BoW dictionary
	BagOfWords bow(codeBookSize_);
	bow.setCodeBook(codeBook_);
	objBoWCode_ = bow.lookUpCodeWords(featureMat_);
}



// Computes a bow hist over a set of point indices
void
FeatureFPFHBoW::GetPtsBoWHist(const pcl::PointIndices::Ptr &points, std::vector<double> &cwhist) const
{
	assert(points->indices.size()!=0);
	cwhist.assign(codeBookSize_,0.0);

	std::vector<int>::iterator ptIdx = points->indices.begin();
	for(; ptIdx!=points->indices.end(); ++ptIdx)
	{
		assert(*ptIdx < (int)objBoWCode_.size());
		cwhist[ objBoWCode_[*ptIdx] ]++;
	}

	// Normalize histogram
	for(std::vector<double>::iterator it=cwhist.begin(); it != cwhist.end(); it++ )
		*it /= points->indices.size();
}


// Computes a bow hist over the whole object
void
FeatureFPFHBoW::GetObjBoWHist( std::vector<double> &cwhist )
{
	cwhist.assign(codeBookSize_,0.0);
	// Eigen::VectorXf codeWordHist = Eigen::VectorXf::Zero(codeBookSize_);
	std::vector<int>::iterator ptIdx = objBoWCode_.begin();
	for(; ptIdx!=objBoWCode_.end(); ++ptIdx)
		cwhist[ objBoWCode_[*ptIdx] ]++;

	// Normalize histogram
	for(std::vector<double>::iterator it=cwhist.begin(); it != cwhist.end(); it++ )
		*it /= objBoWCode_.size();
}


// Gets the bow for a point cloud
int
FeatureFPFHBoW::GetBoWForPoint(uint idx)
{
	assert(idx < objBoWCode_.size());
	return objBoWCode_[idx];
}
