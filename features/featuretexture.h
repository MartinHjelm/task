#ifndef FEATURETEXTURE_H
#define FEATURETEXTURE_H

// Eigen
#include <Eigen/Dense>

// OpenCV
#include <opencv2/core/core.hpp>

class FeatureTexture
{
bool verbose_;

cv::Mat img_;   // Img feference to complete scene
int imgRows_;   // 384
int imgCols_;   // 512

// PCA Projection matrix
Eigen::MatrixXd pcaProjMat_;

// Fisher Vector GMM parameters
int dimension_;   //80 PCA projected dim
int Nclusters_;   //256
Eigen::VectorXd gmmMeans_;
Eigen::VectorXd gmmCovs_;
Eigen::VectorXd gmmPriors_;

// SVM model is 1 against many which means we have k svms for k classes
// svm parameters, i.e., [w_c1,bias_c1;w_c2,bias_c2,...]
Eigen::MatrixXd svmModels_;

void computeDSift(const cv::Mat &img, Eigen::MatrixXd &Xsift, const bool &verbose=false);
void computeFisherVector(const Eigen::MatrixXd &Xsift, Eigen::MatrixXd &Xfisher, const bool &verbose=false);
int predict(const Eigen::MatrixXd &Xfisher, const bool &verbose=false);
void preProcessImg(cv::Mat &img);


public:

FeatureTexture();
void setInputSource(const cv::Mat &img);
double computeTextureLabel(const bool &verbose=false);


};

#endif // FEATURETEXTURE_H
