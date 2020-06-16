#include "featurefpfh.h"

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

featureFPFH::featureFPFH() :
	fpfhRadiusSearch_(0.03),
	normalRadiusSearch_(0.02),
	codeBookSize_(40),
	fpfhProjDim_(20)
{
}


void
featureFPFH::setInputSource(const PC::Ptr &cPtr,const PCN::Ptr &cnPtr)
{
	cloud_ = cPtr; cloudNormals_ = cnPtr;
}

void
featureFPFH::setCodeBook( const std::string &normalRadius, const std::string &fpfhRadius, const std::string &codeBookSize, const std::string &fpfhProjDim )
{
	normalRadiusSearch_ = std::stof(normalRadius);
	fpfhRadiusSearch_ = std::stof(fpfhRadius);
	codeBookSize_ = std::stoi(codeBookSize);
	fpfhProjDim_ = std::stoi(fpfhProjDim);

// Best combination 0.02_0.02_20_20 - 0.02_0.06_40_0 - 0.01_0.05_40_20

	std::string fileParamString = normalRadius+"_"+fpfhRadius+"_"+codeBookSize+"_"+fpfhProjDim;
	std::string fnMeans = "gmmMeans_fpfh_"+fileParamString+".txt";
	std::string fnCovs = "gmmCovs_fpfh_"+fileParamString+".txt";
	std::string fnPriors = "gmmPriors_fpfh_"+fileParamString+".txt";
	std::string fnPCAProjMat = "gmm_projmat_"+fileParamString+".txt";

	// Get codebook and projection matrix file names
	if(!BoostHelperFuns::fileExist(fnMeans))
		throw std::runtime_error("Could not find "+fnMeans);
	if(!BoostHelperFuns::fileExist(fnCovs))
		throw std::runtime_error("Could not find "+fnCovs);
	if(!BoostHelperFuns::fileExist(fnPriors))
		throw std::runtime_error("Could not find "+fnPriors);
	if(fpfhProjDim_ > 0 && !BoostHelperFuns::fileExist(fnPCAProjMat))
		throw std::runtime_error("Could not find "+fnPCAProjMat);

	gmmMeans_ = EigenHelperFuns::readMatrixd(fnMeans);
	gmmCovs_ = EigenHelperFuns::readMatrixd(fnCovs);
	gmmPriors_ = EigenHelperFuns::readVecd(fnPriors);
	if(fpfhProjDim_ > 0)
		pcaProjMat_ = EigenHelperFuns::readMatrixd(fnPCAProjMat);
}


// Computes BoW representation for each point on object
void featureFPFH::cptBoWRepresentation()
{
	computeFPFHfeature();
	encodeFeature();
}


/* Computes the FPFH feature over all points in the point cloud and
 * projects the features using PCA. */
void
featureFPFH::computeFPFHfeature()
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
featureFPFH::encodeFeature()
{
	Eigen::MatrixXd xMaps = Eigen::MatrixXd::Zero(featureMat_.rows(),codeBookSize_);
	for(int iCl = 0; iCl < codeBookSize_; iCl++ )
	{
		Eigen::MatrixXd xBar = featureMat_.rowwise() - gmmMeans_.col(iCl).transpose();
		// Construct cov matrix inverse
		Eigen::VectorXd cov = gmmCovs_.col(iCl).transpose();
		Eigen::VectorXd invcov(cov.size());
		for(uint idx=0; idx < cov.size(); idx++ )
			invcov(idx) = 1./cov(idx);

		Eigen::MatrixXd invCov = Eigen::MatrixXd(invcov.asDiagonal());
		// Eigen::MatrixXd Cov = Eigen::MatrixXd(cov.asDiagonal());

		// N x d  d x d d * N
		Eigen::MatrixXd XCX = Eigen::MatrixXd::Zero(featureMat_.rows(),1);
		for(uint iRow = 0; iRow < featureMat_.rows(); iRow++)
			XCX.row(iRow) = xBar.row(iRow) * invCov * xBar.row(iRow).transpose();

		xMaps.block(0,iCl,xMaps.rows(),1) = -0.5 * XCX.col(0);
		xMaps.col(iCl).array() += gmmPriors_.segment(iCl,1).array().log().sum() - 0.5 * cov.array().log().sum();
	}

	// Compute max of map values
	Eigen::MatrixXd idxs = Eigen::MatrixXd::Zero(xMaps.rows(),1);
	EigenHelperFuns::rowwiseMinMaxIdx(xMaps,idxs,true);
	// Copy bows into the vector
	objBoWCode_ = std::vector<int>(idxs.rows(),0);
	for(uint idx=0; idx < idxs.rows(); idx++)
	{
		objBoWCode_[idx] = (int)idxs(idx,0);
	}
}



// Computes a bow hist over a set of point indices
void
featureFPFH::getPtsBoWHist(const pcl::PointIndices::Ptr &points, std::vector<double> &cwhist) const
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
featureFPFH::getObjBoWHist( std::vector<double> &cwhist )
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
featureFPFH::getBoWForPoint(uint idx)
{
	assert(idx < objBoWCode_.size());
	return objBoWCode_[idx];
}
