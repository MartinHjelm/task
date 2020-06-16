#ifndef BOOSTHELPERFUNS_H
#define BOOSTHELPERFUNS_H

// std
#include <string>
#include <vector>

// BOOST
#include <boost/filesystem.hpp>

class BoostHelperFuns
{
public:
  BoostHelperFuns();

// BOOST
  static bool fileExist(const std::string &fileName);
  static void getListOfFilesInDir(const boost::filesystem::path& root, const std::string &ext, std::vector<boost::filesystem::path> &ret, const bool fullPath=false);
  static void getListOfFilesInDir(const boost::filesystem::path& root, const std::vector<std::string>& ext, std::vector<boost::filesystem::path> &ret, const bool fullPath=false);
  static void getListOfFilesInDirWithName(const boost::filesystem::path& root, const std::string& searchStr, const std::string &ext, std::vector<boost::filesystem::path> &ret);


};


#endif // BOOSTHELPERFUNS_H