
#include <boosthelperfuns.h>


/**************** BOOST SPECIFIC HELPER FUNCTIONS ****************/

BoostHelperFuns::BoostHelperFuns() {}


bool
BoostHelperFuns::fileExist(const std::string &fileName)
{
  return boost::filesystem::exists( fileName );
}



/* Returns the filenames of all files that have the specified extension
 * in the specified directory and all subdirectories
 */
void
BoostHelperFuns::getListOfFilesInDir(const boost::filesystem::path& root, const std::vector<std::string>& ext, std::vector<boost::filesystem::path> &ret, const bool fullPath)
{
  namespace fs = ::boost::filesystem;
  if (!fs::exists(root))
  {
    printf("Directory not found!");
    return;
  }

  if (fs::is_directory(root))
  {
    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {
      //std::cout << it->path().filename() << std::endl;
      if (fs::is_regular_file(*it))
      {
        bool hasExt = false;
        for(int idx = 0; idx < ext.size(); idx++)
          if(it->path().extension() == ext[idx])
          {
            hasExt = true;
            break;
          }
        
        if(hasExt)
        {
          if(fullPath)
            ret.push_back(it->path().string());
          else
            ret.push_back(it->path().filename());
        }
      }
      ++it;
    }
    std::sort(ret.begin(), ret.end());  
  }
}



void
BoostHelperFuns::getListOfFilesInDir(const boost::filesystem::path& root, const std::string &ext, std::vector<boost::filesystem::path> &ret, const bool fullPath)
{
  std::vector<std::string> exts = {ext};
  BoostHelperFuns::getListOfFilesInDir(root, exts, ret, fullPath);
}


void
BoostHelperFuns::getListOfFilesInDirWithName(const boost::filesystem::path& root, const std::string& searchStr, const std::string &ext, std::vector<boost::filesystem::path> &ret)
{
  namespace fs = ::boost::filesystem;
  if (!fs::exists(root)) return;

  if (fs::is_directory(root))
  {
    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;
    while(it != endit)
    {
      //std::cout << it->path().filename() << std::endl;
      if (fs::is_regular_file(*it) and it->path().extension() == ext and (it->path().filename().string()).find(searchStr) != std::string::npos )
      {
        ret.push_back(it->path().filename());
      }
      ++it;
    }
  }
  std::sort(ret.begin(), ret.end());
}


