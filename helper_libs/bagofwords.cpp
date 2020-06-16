#include <bagofwords.h>
#include <kmeans.h>
#include <kmedoids.h>
#include <eigenhelperfuns.h>
#include <myhelperfuns.h>


BagOfWords::BagOfWords() :
  codeBookSize_(10)
{}


BagOfWords::BagOfWords(const int &size) :
  codeBookSize_(size)
{}


double 
BagOfWords::computeCodeBook(const Eigen::MatrixXd &X, const int &kmIters, const int &kmReTrials)
{
  double kmVal = 0.0;
  kMeans km(codeBookSize_,kmIters);
  //kMedoids km(codeBookSize_,kmIters);
  kmVal = km.computeK(X,kmIters,kmReTrials); 
  codeBook_ = km.centroids;
  return kmVal;
}


int
BagOfWords::lookUpCodeWord(const Eigen::VectorXd &vec)
{
  double dBest = INFINITY;
  int idx = 0;
  double d = 0.0;
  for(int iRow=0; iRow!=codeBook_.rows(); iRow++)
  {
    //d = EigenHelperFuns::histDistKernel(codeBook_.row(iRow),vec,intervals);
    // std::cout << vec << std::endl<< std::endl;
    d = EigenHelperFuns::manhattanDist(codeBook_.row(iRow),vec);
    if(d<dBest)
    {
      dBest = d;
      idx = iRow;
    }    
  }

  return idx;
}


std::vector<int>
BagOfWords::lookUpCodeWords(const Eigen::MatrixXd &X)
{
  std::vector<int> codes(X.rows(),0);
  for(int iRow=0; iRow!=X.rows(); iRow++)
  {
    Eigen::VectorXd vec = X.row(iRow);
    codes[iRow] = lookUpCodeWord(vec);
  }
  return codes;
}
