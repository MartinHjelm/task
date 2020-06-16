#include "featurecolorquantization.h"

// std
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>

// Open CV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

// F's segmentation algorithm
#include <cstdio>
#include <cstdlib>
#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

#include "myhelperfuns.h"
#include "boosthelperfuns.h"
#include "pclhelperfuns.h"
#include "eigenhelperfuns.h"

FeatureColorQuantization::FeatureColorQuantization() : printMsg(true), N_comps(0), N_colors(0)
{}

void
FeatureColorQuantization::setInputSource (const std::string &rawfileName, const cv::Mat &segmentedImg, const std::vector<std::vector<int> > &pcPxPos, int &offset, PC::Ptr &cloud)
{
    cloud_ = cloud;
    rawfileName_ = rawfileName;
    offset_ = offset;
    // Copy segmented input img
    segmentedImg.copyTo(img_);
    readQuantColorChart();

    pcPixelPos = pcPxPos;
    cIndices.resize( segmentedImg.rows , std::vector<int>( segmentedImg.cols , 0 ) );
}



/* Quantizes each component of the Felenzwald image segmentation algorithm into
 * a specific color. Each color is assigned to its closests neighbour in the
 * color chart. The main color of each segment is the color most pixels gets
 * assigned too in the segement. */
void
FeatureColorQuantization::colorQuantize ()
{
    /* Segment image using F's graph cut segmentation algorithm.
   * The algorithm outputs a segemented image where each segment
   * has a specific component number.
   */
    runGraphCutSegmentation();

    // Resest pcColorIdx
    pcColorChartIdx.clear();
    colorCodesMat = Eigen::MatrixXi::Zero(img_.rows,img_.cols);

    // Convert to float & CIE Lab color representation since it makes
    // euclidian color distances possible.
    img_.convertTo(img_, CV_32FC3, 1.0/255.0);
    cv::cvtColor(img_, img_, cv::COLOR_BGR2Lab);
    N_colors = cqChart.rows;

    /* Classify each segmented component as belonging to the color in
     * the color chart that most pixels in the segment belongs to.
    */
//    MyHelperFuns::printString("Collecting votes", printMsg);

    // Collect votes for each color for each component
    cv::Vec3f pixVal,chartVal;
    std::vector< std::vector<int> > chartVotes(N_comps, std::vector<int>(N_colors));
    for(int ii=0; ii!=img_.rows; ii++)
    {
        for(int jj=0; jj!=img_.cols; jj++)
        {
            pixVal = img_.at<cv::Vec3f>(ii,jj);
            int colorBest = 0;
            float dist = 0, distBest = 100000.0;
            for(int i_color = 0; i_color != N_colors; i_color++ )
            {
                chartVal = cqChart.at<cv::Vec3f>(i_color,0);
                dist = cv::norm(cqChart.at<cv::Vec3f>(i_color,0)-pixVal);
                if( dist < distBest )
                {
                    distBest = dist;
                    colorBest = i_color;
                }
            }
            chartVotes[cIndices[ii][jj]][colorBest]++;
        }
    }

//    MyHelperFuns::printString("Computing major color", printMsg);

    // Compute the major color for each component
    std::vector<int> clrIdx(N_comps,0);
    for(int i_comp=0; i_comp!=N_comps;i_comp++)
    {
        int N_votes = 0;
        for(int i_color = 0; i_color != N_colors; i_color++ )
        {
            if(chartVotes[i_comp][i_color]>N_votes)
            {
                N_votes = chartVotes[i_comp][i_color];
                clrIdx[i_comp] = i_color;
            }
        }
    }

//    MyHelperFuns::printVector(clrIdx,"Colorindices");

//    MyHelperFuns::printString("Setting component winner", printMsg);
    // Set each pixel in the image to component winner.
    for(int iRow=0; iRow!=img_.rows; iRow++){
        for(int jCol=0; jCol!=img_.cols; jCol++)
        {
            int colorIdx = clrIdx[cIndices[iRow][jCol]];
            colorCodesMat(iRow,jCol) = colorIdx;
//            printf("Colorcode: %i\n",cIndices[iRow][jCol]);
        }
    }

//    MyHelperFuns::printString("Set chart index", printMsg);
    // Set each point in the point cloud's color chart index
    std::vector<std::vector<int> >::iterator iter = pcPixelPos.begin();
    for( ; iter!= pcPixelPos.end(); ++iter)
    {
        int rowIdx = (*iter)[0];
        int colIdx = (*iter)[1];
        int colorIdx = clrIdx[cIndices[rowIdx][colIdx]];
        pcColorChartIdx.push_back(colorIdx);
    }

    cv::cvtColor(img_, img_, cv::COLOR_Lab2BGR);
    img_.convertTo(img_, CV_8UC3, 255.0);
}




/* Loops over a set of points, finds the point's specific quantized color,
 * and adds it to a histogram. */
std::vector<double>
FeatureColorQuantization::computePointHist(const pcl::PointIndices::Ptr &points) const
{
    pcl::PointIndices::Ptr piIDs(new pcl::PointIndices);
    PCLHelperFuns::getOrigInd(cloud_,points, piIDs );
    std::vector<double> colorhist(N_colors,0.0);
    int rowOffset = offset_ / 640;
    int colOffset = offset_ % 640;

    for(std::vector<int>::const_iterator iter = piIDs->indices.begin(); iter!=piIDs->indices.end(); ++iter)
    {
        int row = ( *iter / 640 ) - rowOffset;
        int col = ( *iter % 640 ) - colOffset;
//        printf("Row %i, Col %i\n",row,col); fflush(stdout);
        int colorCode = colorCodesMat(row,col);
//        printf("Colorcode %i\n",colorCode); fflush(stdout);
        // double cid = pcColorChartIdx[colorCode];
        colorhist[colorCode]++;
    }

    // Normalize
    if(points->indices.size()>0)
    {
        for(uint idx = 0; idx != colorhist.size(); idx++ )
            colorhist[idx] = colorhist[idx]/points->indices.size();
    }

    return colorhist;
}


std::vector<double>
FeatureColorQuantization::computePointHist2(const pcl::PointIndices::Ptr &points) const
{
    std::vector<double> colorhist(N_colors,0.0);
    int row, col;
    int rowOffset = offset_ / 640;
    int colOffset = offset_ % 640;

   for(PC::const_iterator iter=cloud_->begin(); iter!=cloud_->end(); ++iter)
    {
        row = ( iter->ID / 640 ) - rowOffset;
        col = ( iter->ID % 640 ) - colOffset;

        int colorCode = colorCodesMat(row,col);
        colorhist[colorCode]++;
    }

    // Normalize
    if(points->indices.size()>0)
    {
        for(uint idx = 0; idx != colorhist.size(); idx++ )
            colorhist[idx] = colorhist[idx]/points->indices.size();
    }

    return colorhist;
}



void
FeatureColorQuantization::imgCQ2PC()
{
    V3f cqVals(0,0,0);
    int row, col;
    int rowOffset = offset_ / 640;
    int colOffset = offset_ % 640;

    cv::Mat cqImg, colorChart;
    getColorQuantizedImg(cqImg);

    cvtColor(cqChart, colorChart, cv::COLOR_Lab2RGB);
    colorChart.convertTo(colorChart, CV_8UC3, 255.0);

    for(PC::const_iterator iter=cloud_->begin(); iter!=cloud_->end(); ++iter)
    {
        row = ( iter->ID / 640 ) - rowOffset;
        col = ( iter->ID % 640 ) - colOffset;

        // Add color quantization
        int colorCode = colorCodesMat(row,col);
        cqVals(0) = colorChart.at<cv::Vec3b>(colorCode,0)[0];
        cqVals(1) = colorChart.at<cv::Vec3b>(colorCode,0)[1];
        cqVals(2) = colorChart.at<cv::Vec3b>(colorCode,0)[2];
//        EigenHelperFuns::printEigenVec(cqVals,"Color ");
        pcCQVals.push_back(cqVals);
        pcCQIdxs.push_back(colorCode);
    }
}





/* Runs Felzenswalbs graph cut segementation algorithm on the window of the
 * segmented image of the object. */
void
FeatureColorQuantization::runGraphCutSegmentation ()
{
    // Segmentation params
    float sigma = 0.05, k = 150; int min_size = 25;
    image<rgb> *input = loadPPM((rawfileName_+".ppm"));
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &N_comps, cIndices);
    // Remove temporary ppm file
//    if(std::remove( std::string(rawfileName_+".ppm").c_str() ))
//       printf("File did not delete!");
//    savePPM(seg, (rawfileName_+"_s.ppm"));
    //MyHelperFuns::printString("Num of graph cut components "+std::to_string(N_comps), printMsg);
}


std::vector<double>
FeatureColorQuantization::computeEntropyMeanVar(const pcl::PointIndices::Ptr &points) const
{
    int N = points->indices.size();
    double mean = 0;
    double stdev = 0;
    double sumsqrds=0., sum=0.;

    pcl::PointIndices::Ptr piIDs(new pcl::PointIndices);
    PCLHelperFuns::getOrigInd(cloud_,points, piIDs );
    std::vector<double> colorhist(N_colors,0.0);
    int rowOffset = offset_ / 640;
    int colOffset = offset_ % 640;

    for(std::vector<int>::const_iterator iter = piIDs->indices.begin(); iter!=piIDs->indices.end(); ++iter)
    {
        int row = ( *iter / 640 ) - rowOffset;
        int col = ( *iter % 640 ) - colOffset;
        int colorNr = colorCodesMat(row,col);
        colorhist[colorNr]++;
        sum += colorNr;
        sumsqrds += colorNr*colorNr;
    }

    // Compute std
    stdev = std::sqrt( (sumsqrds-((sum*sum)/N))/N );
    // Compute mean
    mean = sum / N;

    // Normalize histogram
    if(points->indices.size()>0)
    {
        for(uint idx = 0; idx != colorhist.size(); idx++ )
            colorhist[idx] = colorhist[idx]/N;
    }

    // Compute the entropy
    double entropy=0.0;
    std::vector<double>::iterator valPtr = colorhist.begin();
    for(; valPtr!=colorhist.end();++valPtr)
    {
        if(*valPtr>1E-12)
            entropy -= *valPtr * std::log(*valPtr);
    }

    std::vector<double> feature;
    feature.push_back(entropy);
    feature.push_back(mean);
    feature.push_back(stdev);

    return feature;
}


/* Creates an image colorized according to the quantization */
void
FeatureColorQuantization::getColorQuantizedImg(cv::Mat &cqImg)
{
    img_.copyTo(cqImg);
    cqImg.convertTo(cqImg, CV_32FC3, 1.0/255.0);
    cv::cvtColor(cqImg, cqImg, cv::COLOR_BGR2Lab);
    for(int iRow=0; iRow!=cqImg.rows; iRow++){
        for(int jCol=0; jCol!=cqImg.cols; jCol++)
        {
            cqImg.at<cv::Vec3f>(iRow,jCol) = cqChart.at<cv::Vec3f>(colorCodesMat(iRow,jCol),0);
        }
    }
    cv::cvtColor(cqImg, cqImg, cv::COLOR_Lab2BGR);
    cqImg.convertTo(cqImg, CV_8UC3, 255.0);
}




/* Reads color chart into a OpenCV Matrix */
void
FeatureColorQuantization::readQuantColorChart ()
{
    std::string chartFile("rgb30.txt");
    assert(BoostHelperFuns::fileExist(chartFile));

    // Read file RGB chart into an image matrix
    // CvMLData mlData;
    // mlData.set_delimiter(' ');
    // mlData.read_csv(chartFile.c_str());
    // const CvMat* tmp = mlData.get_values();
    // cv::Mat csvMat(tmp, true);

    // Read RGB File
    Eigen::MatrixXd eigMat = EigenHelperFuns::readMatrixd("rgb30.txt");
    // std::cout << eigMat << std::endl;
    cv::Mat colorChart(eigMat.rows(),1,CV_32FC3);

    // Copy to cv matrix
    for (int i = 0; i < colorChart.rows; i++) {
        colorChart.at<cv::Vec3f>(i, 0)[0] = eigMat(i,0);
        colorChart.at<cv::Vec3f>(i, 0)[1] = eigMat(i,1);
        colorChart.at<cv::Vec3f>(i, 0)[2] = eigMat(i,2);
    }
    // std::cout << eigMat << std::endl;
    // Convert to correct float and lab representation
    colorChart.convertTo(colorChart, CV_32FC3, 1.0/255.0);
    cv::cvtColor(colorChart, cqChart, cv::COLOR_RGB2Lab);
}
