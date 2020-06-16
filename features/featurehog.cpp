#include "featurehog.h"


#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"


// Helper files
#include "pca.h"
#include "kmeans.h"
#include "bagofwords.h"
#include "opencvhelperfuns.h"
#include "boosthelperfuns.h"
#include "eigenhelperfuns.h"


featureHOG::featureHOG() : codeBookSize_(40)
{

    std::string codeBookfName = "codebook_40_hog.txt";

    // Get codebook and projection matrix file names
    if(!BoostHelperFuns::fileExist(codeBookfName))
        throw std::runtime_error("Could not find "+codeBookfName);

//    codeBookSize_ = EigenHelperFuns::readMatrixd(codeBookfName);
    Eigen::MatrixXd codeBook = EigenHelperFuns::readMatrixd(codeBookfName);

    // Create and init BoW dictionary
    bow = BagOfWords(40);
    bow.setCodeBook(codeBook);

}

void
featureHOG::setInputSource(const cv::Mat &img)
{
  img_ = img; // Reference
}

std::vector<double>
featureHOG::getFeature()
{
    return hogHist_;
}

void
featureHOG::appendFeature(std::vector<double> &toVec)
{
    toVec.insert(toVec.end(), hogHist_.begin(), hogHist_.end());
}



void featureHOG::compute()
{
    computeHOG(img_,hogMat_);
    std::vector<int> code = bow.lookUpCodeWords(hogMat_);

    hogHist_ = std::vector<double>(codeBookSize_,0.);
    for(uint i = 0; i!=code.size(); i++)
    {
        hogHist_[code[i]]++;
    }
    // Normalize
    for(int i = 0; i!=codeBookSize_; i++)
    {
        hogHist_[i] /= (float)code.size();
    }

//    print to check
//    double sum = 0.0;
//    for(int i = 0; i!=codeBookSize_; i++)
//    {
//        sum += hogHist_[i];
//        std::cout <<  hogHist_[i] << std::endl;
//    }
//    std::cout << "HOG SUM: " << sum << std::endl;
}







void
featureHOG::computeHOG (cv::Mat &img, Eigen::MatrixXd &hogMat)
{
    // HOG Parameters
    int blockSize = 16;
    int cellSize = 8;
    int blockStride = 8;
    int rowPadding = 0;
    int colPadding = 0;

    // printf("Resizing image\n");
    // Resize img so that HOG descriptor division of blocks works out
    if(img.rows%blockSize>0) rowPadding = blockSize - img.rows%blockSize;
    if(img.cols%blockSize>0) colPadding = blockSize - img.cols%blockSize;
    cv::resize(img, img, cv::Size(img.cols+colPadding,img.rows+rowPadding) );

    // win_size – Detection window size. Align to block size and block stride.
    // block_size – Block size in pixels. Align to cell size. Only (16,16) is supported for now.
    // block_stride – Block stride. It must be a multiple of cell size.
    // cell_size – Cell size. Only (8, 8) is supported for now.
    // nbins – Number of bins. Only 9 bins per cell are supported for now.
    // win_sigma – Gaussian smoothing window parameter.
    // threshold_L2hys – L2-Hys normalization method shrinkage.
    // gamma_correction – Flag to specify whether the gamma correction preprocessing is required or not.
    // nlevels – Maximum number of detection window increases

    // printf("Computing HOG");
    cv::HOGDescriptor d;
    d.winSize = cv::Size(img.cols,img.rows); // Window size as big as the image
    d.blockSize = cv::Size(blockSize,blockSize);
    d.cellSize = cv::Size(cellSize,cellSize);
    d.blockStride = cv::Size(blockStride,blockStride);
    // cv::Size(128,64), //winSize default for human detection
    // cv::Size(16,16), //blocksize
    // cv::Size(8,8), //blockStride,
    // cv::Size(8,8), //cellSize,
    // 9, //nbins,
    // 0, //derivAper,
    // -1, //winSigma,
    // 0, //histogramNormType,
    // 0.2, //L2HysThresh,
    // 0 //gammal correction,
    // //nlevels=64
    //);

    // void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
    //                             Size winStride, Size padding,
    //                             const vector<Point>& locations) const
    std::vector<float> descriptorsValues;
    std::vector<cv::Point> locations;
    d.compute( img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations);

    // std::cout << "HOG descriptor size is " << d.getDescriptorcv::Size() << std::endl;
    // std::cout << "img dimensions: " << img.cols << " width x " << img.rows << " height" << std::endl;
    // std::cout << "Found " << descriptorsValues.cv::Size() << " descriptor values" << std::endl;
    // std::cout << "Nr of locations specified : " << locations.cv::Size() << std::endl;

    int Nhists = descriptorsValues.size()/9;
    int histSize = 9;
    hogMat = Eigen::MatrixXd::Zero(Nhists,histSize);

    for(int i_hist=0;i_hist!=Nhists;i_hist++)
    {
        double histSum = 0;
        for(int i_val=0;i_val!=histSize;i_val++)
        {
            histSum += descriptorsValues[i_val+histSize*i_hist];
            hogMat(i_hist,i_val) = descriptorsValues[i_val+histSize*i_hist];
        }
        // std::cout << "Histogram sum: " << histSum << std::endl;
    }


}
