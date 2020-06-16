#include "opencvhelperfuns.h"

// std
#include <iostream>

// OpenCV
#include <opencv2/imgproc.hpp> 



OpenCVHelperFuns::OpenCVHelperFuns(){};



void
OpenCVHelperFuns::equalizeIntensity(const cv::Mat& inputImage, cv::Mat &outPutImage)
{
    if(inputImage.channels() >= 3)
    {
        cv::Mat ycrcb;
        cv::cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb,channels);
        cv::equalizeHist(channels[0], channels[0]);
        cv::merge(channels,ycrcb);
        cv::cvtColor(ycrcb,outPutImage,CV_YCrCb2BGR);

        return;
    }
}



std::string
OpenCVHelperFuns::type2str(int type) {
  	std::string r;

  	uchar depth = type & CV_MAT_DEPTH_MASK;
  	uchar chans = 1 + (type >> CV_CN_SHIFT);

  	switch ( depth )
  	{
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}



void
OpenCVHelperFuns::printTypeInfo(const std::string &varName, const cv::Mat &A)
{
    std::cout << varName << ":\t " << type2str(A.type()) << " " << A.depth()  << " " << A.channels() << " " << A.size() << std::endl;
}



void
OpenCVHelperFuns::printChannelMats(const cv::Mat &A)
{

  	for (int k = 0; k < 3; k++)
  	{
    	std::cout << "[";
        for (int i = 0; i < A.rows; i++)
        {
        	for (int j = 0; j < A.cols; j++)
          	{
        		// cv::Vec3b cVals = A.at<cv::Vec3b>(i,j);
        //std::cout << cVals[k] << ", ";
            std::cout << A.at<cv::Vec3f>(i,j)[k];
            if(j+1<A.cols && i!=A.rows)
              std::cout << ", ";
      }
      if(i+1<A.rows)
        std::cout << "," << std::endl;
	}
	std::cout << "]" << std::endl << std::endl;
	}
}



/* Sets the pixels, given by the indices argument, to zero.*/
void
OpenCVHelperFuns::setIdx2Zero( cv::Mat &img, const std::vector<std::vector<int> > &indices, const bool &negative )
{

	if(negative) // Set to black everyting in the indices vector
  	{
    	std::vector<std::vector<int> >::const_iterator iter = indices.begin();
    	for(; iter!=indices.end(); iter++)
    	{
      		int row = (*iter)[0];
      		int col = (*iter)[1];
      		for(int colorIdx = 0; colorIdx!=3; colorIdx++)
        		img.at<cv::Vec3b>(row,col)[colorIdx] = 0;
		}
	}
  	else  // Set to black everyting NOT in the indices vector
  	{
    	cv::Mat imgCpy = cv::Mat::zeros(img.size(),CV_8UC3);
    	std::vector<std::vector<int> >::const_iterator iter = indices.begin();
    	for(; iter!=indices.end(); iter++)
    	{
      		int row = (*iter)[0];
      		int col = (*iter)[1];
      		for(int colorIdx = 0; colorIdx!=3; colorIdx++)
      		{
        		imgCpy.at<cv::Vec3b>(row,col) = img.at<cv::Vec3b>(row,col);
    		}
		}
    imgCpy.copyTo(img); // Overwrite old image with new image
	}
}
