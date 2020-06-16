#include <featuretexture.h>

// std
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <eigenhelperfuns.h>

// VLFeat
// The VLFeat header files need to be declared external.
extern "C" {
#include <vlfeat/vl/dsift.h>
#include <vlfeat/vl/fisher.h>
#include <vlfeat/vl/generic.h>
#include <vlfeat/vl/gmm.h>
#include <vlfeat/vl/svm.h>
}

FeatureTexture::FeatureTexture()
    : verbose_(false), imgRows_(384), imgCols_(512), dimension_(80),
      Nclusters_(256) {
  // GMM parameters
  gmmMeans_ = EigenHelperFuns::readVecd("gmmMeans.txt");
  gmmCovs_ = EigenHelperFuns::readVecd("gmmCovs.txt");
  gmmPriors_ = EigenHelperFuns::readVecd("gmmPriors.txt");

  // PCA projection matrix
  pcaProjMat_ = EigenHelperFuns::readMatrixd("pcaProjMat.txt");

  // SVM models
  svmModels_ = EigenHelperFuns::readMatrixd("svmModels.txt");
}

void FeatureTexture::setInputSource(const cv::Mat &img) {
  img_ = img; // Reference
}

double FeatureTexture::computeTextureLabel(const bool &verbose) {
  // Resize, rescale, crop, convert to grayscale, etc., i.e. make ready for SIFT
  cv::Mat img;
  cv::cvtColor(img_, img, cv::COLOR_BGR2GRAY);

  //    img =
  //    cv::imread("/Users/martinhjelm/Dropbox/Code/DATASETS/vision/test/fabric_moderate_001_new.jpg",0);
  preProcessImg(img);

  //    cv::imshow( "Original", img_ );
  //    cv::imshow( "Processed", img );
  //    cv::waitKey(0);

  // Compute dense SIFT representation of image dsift always returns the exact
  // same feature size
  Eigen::MatrixXd Xsift;
  computeDSift(img, Xsift, verbose);

  // Project Matrix onto principal axis
  if (verbose)
    std::cout << "\n\033[92m"
              << "# Projecting onto principal axes...Done."
              << "\033[0m\n"
              << std::flush;
  Eigen::MatrixXd Xpca = Xsift * pcaProjMat_;

  // Compute Fisher Vector for image
  Eigen::MatrixXd Xfisher =
      Eigen::MatrixXd::Zero(1, 2 * dimension_ * Nclusters_);
  computeFisherVector(Xpca, Xfisher, verbose);

  // Classify using SVM
  int label = predict(Xfisher, verbose);

  return label;
}

void FeatureTexture::preProcessImg(cv::Mat &img) {
  // If image size is lesser than the default we just pad it with a zero border.
  // int top, int bottom, int left, int right,

  //    if(img.rows < imgRows_)
  //    {
  //        int borderTop = (imgRows_ - img.rows) / 2;
  //        int borderBottom = imgRows_ - img.rows - borderTop;
  //        cv::copyMakeBorder(img,img,borderTop,borderBottom,0,0,cv::BORDER_CONSTANT,cv::Scalar(0));
  //    }

  //    if(img.cols < imgCols_)
  //    {
  //        int borderRight = (imgCols_ - img.cols) / 2;
  //        int borderLeft = imgCols_ - img.cols - borderRight;
  //        cv::copyMakeBorder(img,img,0,0,borderLeft,borderRight,cv::BORDER_CONSTANT,cv::Scalar(0));
  //    }

  // If size still is bigger than default we resize it
  //    if(img.cols != imgCols_ && img.rows != imgRows_ )
  //    {
  //        cv::Size size(imgCols_,imgRows_);
  //        cv::resize(img,img,size);
  //    }

  // Do histogram equalization
  //    cv::equalizeHist( img, img );
}

void FeatureTexture::computeDSift(const cv::Mat &img, Eigen::MatrixXd &Xsift,
                                  const bool &verbose) {

  if (verbose)
    std::cout << "\n\033[92m"
              << "# Computing Dense SIFT for all images..."
              << "\033[0m" << std::flush;

  // transform image in cv::Mat to float vector
  std::vector<float> imgvec;
  imgvec.reserve(img.rows * img.cols);
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      imgvec.push_back(img.at<unsigned char>(i, j) / 255.0f);
    }
  }

  // create filter
  VlDsiftFilter *vlf = vl_dsift_new_basic(img.rows, img.cols, 4, 8);

  // call processing function of vl
  vl_dsift_process(vlf, &imgvec[0]);

  // std::cout << "Number of keypoints: " << vl_dsift_get_descriptor_size(vlf)
  // << std::flush << std::endl;
  // std::cout  << "Descriptor size: " << vl_dsift_get_keypoint_num(vlf) <<
  // std::flush << std::endl;

  int descSize = vl_dsift_get_descriptor_size(vlf);
  int Nkeypoints = vl_dsift_get_keypoint_num(vlf);
  std::cout << vl_dsift_get_keypoint_num(vlf) << std::endl;

  // Copy descriptors to eigen matrix
  float *descArray = (float *)vl_malloc(sizeof(float) * descSize * Nkeypoints);
  descArray = (float *)vl_dsift_get_descriptors(vlf);

  // For each descriptor
  // For each descriptor value
  Xsift = Eigen::MatrixXd::Zero(Nkeypoints, descSize);
  for (int kpIdx = 0; kpIdx < Nkeypoints; kpIdx++)
    for (int descIdx = 0; descIdx < descSize; descIdx++)
      Xsift(kpIdx, descIdx) = descArray[kpIdx * descSize + descIdx];

  // Free memory
  vl_dsift_delete(vlf);

  if (verbose)
    std::cout << "\033[92m"
              << "Done."
              << "\033[0m\n"
              << std::flush;
}

void FeatureTexture::computeFisherVector(const Eigen::MatrixXd &Xsift,
                                         Eigen::MatrixXd &Xfisher,
                                         const bool &verbose) {

  if (verbose)
    std::cout << "\n\033[92m"
              << "# Encoding Fisher vectors..."
              << "\033[0m" << std::flush;

  // Map gmm parameters eigen vectors to vlfeat variables
  double *gmmMeans = (double *)vl_malloc(sizeof(double) * gmmMeans_.size());
  Eigen::Map<Eigen::VectorXd>(gmmMeans, gmmMeans_.size()) = gmmMeans_;

  double *gmmCovs = (double *)vl_malloc(sizeof(double) * gmmCovs_.size());
  Eigen::Map<Eigen::VectorXd>(gmmCovs, gmmCovs_.size()) = gmmCovs_;

  double *gmmPriors = (double *)vl_malloc(sizeof(double) * gmmPriors_.size());
  Eigen::Map<Eigen::VectorXd>(gmmPriors, gmmPriors_.size()) = gmmPriors_;

  double *Xs =
      (double *)vl_malloc(sizeof(double) * Xsift.rows() * Xsift.cols());
  Eigen::Map<Eigen::MatrixXd>(Xs, Xsift.rows(), Xsift.cols()) = Xsift;

  // Encode
  vl_size dimension = dimension_;
  vl_size numClusters = Nclusters_;
  double *enc =
      (double *)vl_malloc(sizeof(double) * 2 * dimension * numClusters);
  vl_fisher_encode(enc, VL_TYPE_DOUBLE, gmmMeans, dimension, numClusters,
                   gmmCovs, gmmPriors, Xs, Xsift.rows(),
                   VL_FISHER_FLAG_IMPROVED);

  // Copy to Eigen mat
  for (uint iVal = 0; iVal < 2 * dimension * numClusters; iVal++)
    Xfisher(0, iVal) = enc[iVal];

  vl_free(enc);
  delete gmmMeans;
  delete gmmCovs;
  delete gmmPriors;

  if (verbose)
    std::cout << "\n\033[92m"
              << "Done."
              << "\033[0m\n"
              << std::flush;
}

int FeatureTexture::predict(const Eigen::MatrixXd &X, const bool &verbose) {

  if (verbose)
    std::cout << "\n\033[92m"
              << "# Classifying test data"
              << "\033[0m\n"
              << std::flush;

  // Compute s = w*x.T for all data -> S = W * X.T. Where W is the weight matrix
  // for each models and X is the data matrix.
  // S(k x N) = W(k x M) * X(N x M).T
  Eigen::MatrixXd scores(svmModels_.rows(), X.rows());
  scores = svmModels_.block(0, 0, svmModels_.rows(), svmModels_.cols() - 1) *
           X.transpose();

  // Add bias, s += b
  for (int label = 0; label < svmModels_.rows(); label++)
    scores.row(label).array() += svmModels_(label, svmModels_.cols() - 1);

  if (verbose) {
    std::cout << "Printing SVM scores...\n" << std::flush;
    std::cout << scores << std::endl;
  }

  // For each column in S(k x N) find the max value and corresponding row index
  // which indicates the label.
  // idxs(1 x N)
  Eigen::MatrixXd idxs = Eigen::MatrixXd::Zero(1, X.rows());
  EigenHelperFuns::colwiseMinMaxIdx(scores, idxs);

  int labelId = (int)idxs(0, 0);

  if (verbose) {
    std::cout << idxs << std::endl;
    std::vector<std::string> labelNames = {
        "fabric", "glass", "carton", "porcelain", "metal", "plastic", "wood"};
    std::cout << "Predicted material: " << labelNames[labelId] << std::endl;
  }

  return labelId;
}
