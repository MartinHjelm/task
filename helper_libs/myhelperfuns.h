#ifndef MYHELPERFUNS_H
#define MYHELPERFUNS_H

// std
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

// Boost
#include <boost/lexical_cast.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/range/algorithm.hpp>

class MyHelperFuns
{
public:
  MyHelperFuns();

  // STD functions
  template<class T> static void printVector(const std::vector<T> &vec, const std::string &vecName="Vector: ");
  template<class T> static void print2DVector(const std::vector< std::vector<T> > &vec, const std::string &vecName="Vector: " );
  static inline void printString(const std::string &str, const bool printOn) { if(printOn) std::cout << str << std::endl; }

  template<class T> static T maxValVector(const std::vector<T> &vec);
  template<class T> static T minValVector(const std::vector<T> &vec);
  template<class T> static int minIdxOfVector(const std::vector<T> &vec);
  template<class T> static int maxIdxOfVector(const std::vector<T> &vec);
  template<class T> static std::string toString(T val);
  template<class T> static std::vector<size_t> sort_indexes(const std::vector<T> &v, const bool sortOrder=true);
  template<class T> static bool greaterThan(const T &v1, const T &v2);
  template<class T> static bool lessThan(const T &v1, const T &v2);

  template<class T> static int writeVecToFile2(const std::string &fileName, const std::vector<T> &vec, const std::string &delimeter="/t");


  static inline int randIdx(const int &Nmax){ return std::rand() % Nmax; }

  static void writeVecToFile(const std::string &fileName, const std::vector<double> &vec, const std::string &delimeter="/t");
  static void readVecFromFile(const std::string &fileName, std::vector<double> &vec);
  static int writeStringToFile(const std::string &fileName, const std::string &str);
  static std::vector<int> getColor(const int &colorID);
  
  static void normalizeVec(std::vector<double> &vec);
  static void vec2Hist(const std::vector<double> &vec, const int &bins, std::vector<double> &hist);
  static double manhattanDist(const std::vector<double> &vec1, const std::vector<double> &vec2);
};



/** HEADER TEMPLATE FUNCTIONS **/


template<class T> void
MyHelperFuns::printVector(const std::vector<T> &vec, const std::string &vecName)
{
  if(vec.size()==0)
  {
    std::cout << vecName << " []" << std::endl;
    return;
  }

  std::cout << vecName << " [";
  typename std::vector<T>::const_iterator it;
  for(it = vec.begin(); it!=vec.end()-1; it++){
    std::cout << *it << ", ";
  }
  std::cout << vec[vec.size()-1] << "]" << std::endl;
}

template<class T> void
MyHelperFuns::print2DVector(const std::vector< std::vector<T> > &vec, const std::string &vecName )
{
  if(vec.size()==0)
  {
    std::cout << vecName << " []" << std::endl;
    return;
  }

  std::cout << vecName << " [";
  typename std::vector< std::vector<T> >::const_iterator row;
  typename std::vector<T>::const_iterator col;
  for (row = vec.begin(); row != vec.end(); row++) {
    for (col = row->begin(); col != row->end(); col++) {
      std::cout << *col << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;  
}

template<class T> T
MyHelperFuns::minValVector(const std::vector<T> &vec )
{
  return *std::min_element(vec.begin(),vec.end());
}

template<class T> T
MyHelperFuns::maxValVector(const std::vector<T> &vec )
{
  return *std::max_element(vec.begin(),vec.end());
}

template<class T> int
MyHelperFuns::minIdxOfVector(const std::vector<T> &vec)
{
  return std::min_element(vec.begin(), vec.end()) - vec.begin();
}

template<class T> int
MyHelperFuns::maxIdxOfVector(const std::vector<T> &vec)
{
  return std::max_element(vec.begin(), vec.end()) - vec.begin();
}

template<class T> std::string
MyHelperFuns::toString(T val)
{
  //return boost::lexical_cast<std::string>(val);
  std::ostringstream ss;
  ss << val;
  return ss.str();
}

template<class T>
std::vector<size_t> MyHelperFuns::sort_indexes(const std::vector<T> &v, const bool sortOrder) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    if(sortOrder) // Ascending
    {
        // sort indexes based on comparing values in v
       std::sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
        // std::sort(idx.begin(), idx.end(),
           // bind(&v[*boost::lambda::_1x], boost::lambda::_1)  < bind(&v[*boost::lambda::_2], boost::lambda::_2) );
    }
    else
    {
        // sort indexes based on comparing values in v
        // std::sort(idx.begin(), idx.end(),
        // greaterThan<T>);
        std::sort(idx.begin(), idx.end(),
           [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});        
    }

  return idx;
}

template<typename T>
bool MyHelperFuns::greaterThan(const T &v1, const T &v2)
{
    return v1 > v2;
}

template<typename T>
bool MyHelperFuns::lessThan(const T &v1, const T &v2)
{
    return v1 < v2;
}


template<class T> 
int MyHelperFuns::writeVecToFile2(const std::string &fileName, const std::vector<T> &vec, const std::string &delimeter)
{
  std::ofstream file(fileName.c_str(), std::ios::out| std::ios::app);
  if (file.is_open())
  {
    for(typename std::vector<T>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter)
      if(iter+1 != vec.end())
        file << (*iter) << delimeter;
      else 
        file << (*iter);
  }
  file << std::endl;
  file.close();  
  return 1;
}




#endif // MYHELPERFUNS_H
