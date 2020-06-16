#include "featurecolorhist.h"
#include "opencvhelperfuns.h"
#include "opencv2/imgproc/imgproc.hpp"
FeatureColorHist::FeatureColorHist() : bins_(11)
{}

void
FeatureColorHist::setInputSource(const cv::Mat &img)
{
  img_ = img; // Reference
}


void FeatureColorHist::computeFeatureMats()
{
  computeImgGrad(img_,imgGrad1_,1);
  computeImgGrad(img_,imgGrad2_,2);
  computeImgGrad(img_,imgGrad3_,3);
  computeBrightness(img_,brightnessMat_);
}

// Converts between array and matrix position
std::pair<int,int>
FeatureColorHist::arIdx2MatPos(const int &idx, const int &imWidth) const
{
  std::pair<int,int> pos;
  // Row
  pos.first = idx / imWidth;
  // Col
  pos.second = idx % imWidth;

  return pos;
}

// Converts a one channel mat to a std vector.
std::vector<double>
FeatureColorHist::mat2Vec(const cv::Mat &mat) const
{
  OpenCVHelperFuns::printTypeInfo("Con ", mat);

  std::vector<double> vec;
  for(int iRow = 0; iRow!=mat.rows; iRow++)
  {
    // Pointer to the i-th row
    const double* p = mat.ptr<double>(iRow);
    // Copy data to a vector.  Note that (p + mat.cols) points to the
    // end of the row.
    vec.insert(vec.end(), p, p + mat.cols);
  }
  return vec;
}


/****** FILTER FUNS ******/

// Sobel filter
void
FeatureColorHist::computeImgGrad(const cv::Mat &imgIn, cv::Mat &imgOut, const int &gradOrder)
{
  int kernelSz = 3;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  if(gradOrder>2)
    kernelSz = 5;

  if(gradOrder==1)
  {
      cv::Mat imgGray;
      // Apply Gaussian Blur
      cv::GaussianBlur( imgIn, imgOut, cv::Size(3,3),0,0, cv:: BORDER_DEFAULT);
      // Convert the image to grayscale
      cv::cvtColor( imgOut, imgGray, cv::COLOR_BGR2GRAY );

      /// Generate grad_x and grad_y
      cv::Mat grad_x, grad_y;
      cv::Mat abs_grad_x, abs_grad_y;

      /// Gradient X
      //  cv::Scharr( imgOut, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
      cv::Sobel( imgGray, grad_x, ddepth, gradOrder, 0, kernelSz, scale, delta, cv::BORDER_DEFAULT );
      cv::convertScaleAbs( grad_x, abs_grad_x );

      /// Gradient Y
      //  cv::Scharr( imgOut, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
      cv::Sobel( imgGray, grad_y, ddepth, 0, gradOrder, kernelSz, scale, delta, cv::BORDER_DEFAULT );
      cv::convertScaleAbs( grad_y, abs_grad_y );

      /// Total Gradient (approximate)
      cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgOut );
      //cv::threshold(imgOut, imgOut, 125, 0, cv::THRESH_TOZERO );
      // cv::imshow( "GaussianBlur", imgOut );
      // cv::waitKey();
      //cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);

  }
  else if(gradOrder==2)
  {
      cv::Mat imgGray, imgLap;
      // Apply Gaussian Blur
      cv::GaussianBlur( imgIn, imgOut, cv::Size(9,9),0.1,0.1 );
      // cv::GaussianBlur( imgIn, imgOut, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
      // Convert the image to grayscale
      cv::cvtColor( imgOut, imgGray, cv::COLOR_BGR2GRAY );
      // Apply Laplace function
      cv::Laplacian( imgGray, imgLap, ddepth, kernelSz, scale, delta, cv::BORDER_DEFAULT );
      cv::convertScaleAbs( imgLap, imgOut );
      //cv::threshold(imgOut, imgOut, 125, 0, cv::THRESH_TOZERO );
      //cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);
  }
  else if(gradOrder==3)
  {
      cv::Mat imgGray, imgLap;
      // Apply Gaussian Blur
      cv::GaussianBlur( imgIn, imgOut, cv::Size(9,9),0.1,0.1 );
      // cv::bilateralFilter( imgIn, imgOut, 9,100,100);
      cv::convertScaleAbs( imgLap, imgLap );
      // cv::imshow("Orig", imgIn );
      // cv::waitKey();
      // cv::imshow( "GaussianBlur", imgOut );
      // cv::waitKey();

      // Convert the image to grayscale
      cv::cvtColor( imgOut, imgGray, cv::COLOR_BGR2GRAY );
      // Apply Laplace function
      cv::Laplacian( imgGray, imgLap, ddepth, kernelSz, scale, delta, cv::BORDER_DEFAULT );

      /// Generate grad_x and grad_y
      cv::Mat grad_x, grad_y;
      cv::Mat abs_grad_x, abs_grad_y;
      cv::Mat gradMat;

      /// Gradient X
      //  cv::Scharr( imgOut, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
      cv::Sobel( imgGray, grad_x, ddepth, gradOrder, 0, kernelSz, scale, delta, cv::BORDER_DEFAULT );
      // cv::convertScaleAbs( grad_x, abs_grad_x );

      /// Gradient Y
      //  cv::Scharr( imgOut, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
      cv::Sobel( imgGray, grad_y, ddepth, 0, gradOrder, kernelSz, scale, delta, cv::BORDER_DEFAULT );
      // cv::convertScaleAbs( grad_y, abs_grad_y );

      /// Total Gradient (approximate)
      // cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgOut );
      cv::addWeighted( grad_x, 0.5, grad_y, 0.5, 0, gradMat );
        // cv::convertScaleAbs( imgLap, imgOut );

      /* 0: Binary
         1: Binary Inverted
         2: Threshold Truncated
         3: Threshold to Zero
         4: Threshold to Zero Inverted
       */

      // threshold 3order derivatives
      cv::Mat threshMat;
      // std::cout << "M = "<< std::endl << " "  << gradMat << std::endl << std::endl;
      cv::threshold(gradMat, threshMat, -1, 255, cv::THRESH_BINARY_INV );
      threshMat.convertTo(threshMat, CV_8U);
      // cv::convertScaleAbs( gradMat, gradMat );
      // cv::imshow( "Threshold 3Grad", threshMat );
      // cv::waitKey();

      // Get zero crossing of laplace
      cv::Mat zeroCrossMaskMat;
      // cv::imshow( "Threshold 3Grad", imgLap );
      // cv::waitKey();
      getZeroCrossings(10., zeroCrossMaskMat, imgLap);
      // cv::imshow( "Threshold 3Grad", threshMat );
      // cv::waitKey();

      cv::Mat maskMat,img2;
      imgLap.copyTo(maskMat, zeroCrossMaskMat);
      maskMat.copyTo(img2, threshMat);
      // cv::Mat img2 = threshMat.mul(maskMat);
      cv::convertScaleAbs( img2, imgOut );

      //cv::convertScaleAbs( imgLap, imgLap );
      // cv::imshow("Orig Laplace", imgLap );
      // cv::waitKey();
      // cv::imshow( "MultLaplace", imgOut );
      // cv::waitKey();
      //cv::normalize(imgOut, imgOut, 0, 255, cv::NORM_MINMAX);
    }
}


void FeatureColorHist::getZeroCrossings(float threshold, cv::Mat &binary, const cv::Mat &laplace){

  // Create the iterators
  cv::Mat_<float>::const_iterator it = laplace.begin<float>()+laplace.step1();
  cv::Mat_<float>::const_iterator itend = laplace.end<float>();
  cv::Mat_<float>::const_iterator itup = laplace.begin<float>();

  // Binary image initialize to white
  binary = cv::Mat(laplace.size(),CV_8U,cv::Scalar(255));
  cv::Mat_<uchar>::iterator itout = binary.begin<uchar>()+binary.step1();
  cv::Mat_<uchar>::iterator itoutend = binary.end<uchar>();
  // negate the input threshold value
  threshold *= -1.0;

  // std::cout << "Size" << laplace.size() << binary.size() << std::endl;
  // printf("Starting lp iteration.\n" ); std::cout << std::flush;

  // std::cout << (std::distance(it, itend)) << std::endl << std::flush;
  // std::cout << (std::distance(it, itup)) << std::endl << std::flush;

  for ( ; it!= itend; ++it, ++itup, ++itout) {
    if(itout==(itoutend-1)) break;
    // if the product of two adjacent pixels is negative
    // then there is a sign change
    if (*it * *(it-1) < threshold)
      *itout= 0; // horizontal zero-crossing
    else if (*it * *itup < threshold)
      *itout= 0; // vertical zero-crossing
  }
  // printf("Ending lp iteration.\n" ); std::cout << std::flush;

}



// Brightness filter, that is, grayscale intensity..
void
FeatureColorHist::computeBrightness(const cv::Mat &imgIn, cv::Mat &imgOut)
{
  // Convert to HSV
  cv::Mat hsvMat;
  cv::Mat hsvSplitMat[3];
  cv::cvtColor(imgIn, hsvMat, cv::COLOR_RGB2HSV);

  // Split channels
  cv::split(hsvMat,hsvSplitMat);
  hsvSplitMat[2].copyTo(imgOut);
  cv::convertScaleAbs( imgOut, imgOut );
}



/****** HISTOGRAM FUNCTIONS ******/
void
FeatureColorHist::histGrad(const pcl::PointIndices::Ptr &points, const int &gradOrder, std::vector<double> &hist) const
{
//  int bins = 11;
  hist.clear();
  if (gradOrder==0)
    computeGenericHist(brightnessMat_,points,bins_,hist);
  else if(gradOrder==1)
    computeGenericHist(imgGrad1_,points,bins_,hist);
  else if(gradOrder==2)
    computeGenericHist(imgGrad2_,points,bins_,hist);
  else if(gradOrder==3)
      computeGenericHist(imgGrad3_,points,bins_,hist);
}


void
FeatureColorHist::histBrightness(const pcl::PointIndices::Ptr &points, std::vector<double> &hist) const
{
//  int bins = 11;
  hist.clear();
  computeGenericHist(brightnessMat_,points,bins_,hist);
}


/* Given an img, a set of pointindices, bins and a hist vector this function
 *
*/
void
FeatureColorHist::computeGenericHist(const cv::Mat &imgIn, const pcl::PointIndices::Ptr &points, const int &Nbins, std::vector<double> &hist) const
{
  assert(points->indices.size()>0);

  /*First we take all point indices convert them to positions in the image. */
  cv::Mat pointsMat(points->indices.size(),1,CV_8UC1);
  for(uint idx=0; idx!=points->indices.size(); idx++)
  {
    std::pair<int,int> pos = arIdx2MatPos(points->indices[idx],img_.cols);
    assert(pos.first<img_.rows && pos.second<img_.cols);
    pointsMat.at<uchar>(idx,0) = imgIn.at<uchar>(pos.first,pos.second);
  }

  /* Compute the histogram */
  int bins[1] = {Nbins};
  float imgRanges[] = { 0, 256 };
  const float* ranges[] = { imgRanges };
  int channels[] = {0};
  cv::Mat histMat;
  cv::calcHist(&pointsMat,1,channels,cv::Mat(),histMat,1,bins,ranges,true,false);

  // Copy to vector
  for(int i=0;i<histMat.rows;i++)
    hist.push_back(histMat.at<float>(0,i));

  // Normalize by # of elements
  for(uint i=0; i<hist.size(); i++)
    hist[i] /= points->indices.size();
}





// Returns the gradient value for a specific point in cloud
double
FeatureColorHist::gradAtPoint(const int &idx, const int &gradOrder, const int &imwidth) const
{

  std::pair<int,int> pos = arIdx2MatPos(idx,imwidth);
  // std::cout << pos.first << " " << pos.second << " " << imwidth << " " << idx <<std::endl;
  // std::cout << brightnessMat_.rows << " "  << brightnessMat_.cols << '\n';
  if(gradOrder==0)
  {
    return brightnessMat_.at<uchar>(pos.first,pos.second);
  }

  if(gradOrder==1)
  {
      return imgGrad1_.at<uchar>(pos.first,pos.second);
  }

  if(gradOrder==2)
  {
    return imgGrad2_.at<uchar>(pos.first,pos.second);
  }

  if(gradOrder==3)
  {
    return imgGrad3_.at<uchar>(pos.first,pos.second);
  }

  return 0.0;
}


int
FeatureColorHist::binAtPoint(const int &idx, const int &gradOrder, const int &imwidth) const
{
    int binSize = 23; // 255/11

    std::pair<int,int> pos = arIdx2MatPos(idx,imwidth);

    // std::cout << pos.first << " " << pos.second << " " << imwidth << " " << idx <<std::endl;

    if(gradOrder==0)
    {
        int binIdx = ( (int) brightnessMat_.at<uchar>(pos.first,pos.second)) / binSize;
        return (binIdx == bins_) ? binIdx-1 : binIdx;
    }

    if(gradOrder==1)
    {
        int binIdx = ( (int) imgGrad1_.at<uchar>(pos.first,pos.second)) / binSize;
        return (binIdx == bins_) ? binIdx-1 : binIdx;
    }

    if(gradOrder==2)
    {
        int binIdx = ( (int) imgGrad2_.at<uchar>(pos.first,pos.second)) / binSize;
        return (binIdx == bins_) ? binIdx-1 : binIdx;
    }

    if(gradOrder==3)
    {
        int binIdx = ( (int) imgGrad3_.at<uchar>(pos.first,pos.second)) / binSize;
        return (binIdx == bins_) ? binIdx-1 : binIdx;
    }

    return 0;
}
