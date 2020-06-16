#include "myhelperfuns.h"

// std
#include <iostream>
#include <fstream>
#include <stdexcept>

int
MyHelperFuns::writeStringToFile(const std::string &fileName, const std::string &str)
{
  std::ofstream file(fileName.c_str(), std::ios::out| std::ios::app);
  if (file.is_open())
    file << str << "\n";
else
    return 0;
file.close();
return 1;
}

void
MyHelperFuns::writeVecToFile(const std::string &fileName, const std::vector<double> &vec, const std::string &delimeter)
{
  std::ofstream file(fileName.c_str(), std::ios::out| std::ios::app);
  if (file.is_open())
  {
    for(std::vector<double>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
      if(iter+1 != vec.end())
        file << (*iter) << delimeter;
      else 
        file << (*iter);
}
file << std::endl;
file.close();
}


void 
MyHelperFuns::readVecFromFile(const std::string &fileName, std::vector<double> &vec)
{

  std::ifstream file(fileName.c_str());
  std::string line;
  vec.clear();
  if (file.is_open())
  {
    while ( std::getline (file,line) )
    {
       double data = std::atof(line.c_str());
       vec.push_back(data);
   }
   file.close();
}
}


std::vector<int>
MyHelperFuns::getColor(const int &colorID)
{
  //assert(colorID<41);
  int idx = colorID%41;
  int colorArray[41][3] = { {102,0,0}, {217,108,108}, {255,34,0}, {64,29,16}, {140,115,105}, {153,61,0}, {229,122,0}, {255,196,128}, {89,68,45}, {242,222,182}, {229,214,0}, {127,121,32}, {62,89,45}, {217,255,191}, {109,242,61}, {0,166,66}, {182,242,222}, {0,115,92}, {57,230,195}, {32,64,62}, {0,226,242}, {0,145,217}, {89,149,179}, {48,54,64}, {0,24,89}, {191,208,255}, {102,116,204}, {57,65,115}, {34,0,255}, {133,51,204}, {194,153,204}, {238,0,255}, {107,0,115}, {51,0,48}, {230,57,172}, {77,57,68}, {229,0,92}, {127,0,51}, {51,13,23}, {255,191,208}, {178,0,24}};
//  std::vector<std::vector<int> > colors(41,std::vector<int>(3,0));

  std::vector<int> color(3,0);
  color[0]=colorArray[idx][0];
  color[1]=colorArray[idx][1];
  color[2]=colorArray[idx][2];
  return color;
}



void
MyHelperFuns::normalizeVec(std::vector<double> &vec)
{
    double sum = 0.0;
    std::vector<double>::iterator valPtr = vec.begin();
    for(;valPtr!=vec.end(); valPtr++)
        sum += *valPtr;
    for(valPtr = vec.begin(); valPtr!=vec.end(); valPtr++)
        *valPtr /= sum;
}




void
MyHelperFuns::vec2Hist(const std::vector<double> &vec, const int &bins, std::vector<double> &hist)
{
  hist.clear();
  hist.resize(bins,0.0);
  if(vec.empty())
    return;


  double min = minValVector(vec);
  double max = maxValVector(vec);
  max += 1E-10;
  double binSize = (max-min)/bins;

  // Increment
  for(std::vector<double>::const_iterator valPtr = vec.begin(); valPtr!= vec.end(); valPtr++)
  {
    int pos = std::fabs(*valPtr-min)/binSize;
    hist[pos]++;
  }

  // Normalize
  for(std::vector<double>::iterator valPtr = hist.begin(); valPtr!= hist.end(); valPtr++)
  {
    if(*valPtr>1E-10)
      *valPtr /= (float)vec.size();
  }

}

double
MyHelperFuns::manhattanDist(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
  assert(vec1.size()==vec2.size());

  double dist = 0.0;
  for(int idx=0; idx!=vec1.size(); idx++)
    dist += std::fabs(vec1[idx]-vec2[idx]);

  return dist;
}




