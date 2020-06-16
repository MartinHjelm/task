#ifndef OPENCVHELPERFUNS_H
#define OPENCVHELPERFUNS_H

// std
#include <string>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>


class OpenCVHelperFuns
{
public:
  	OpenCVHelperFuns();

  	// OpenCV helper functions
  	static void printChannelMats(const cv::Mat &A);
  	static void printTypeInfo(const std::string &varName, const cv::Mat &A);
  	static std::string type2str(int type);
  	static void setIdx2Zero( cv::Mat &img, const std::vector<std::vector<int> > &indices, const bool &negative = true );
	static void equalizeIntensity(const cv::Mat& inputImage, cv::Mat &outPutImage);
};

#endif // OPENCVHELPERFUNS_H
