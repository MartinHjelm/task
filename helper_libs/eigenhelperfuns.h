#ifndef EIGENHELPERFUNS_H
#define EIGENHELPERFUNS_H

// std
#include <string>
#include <vector>

// Eigen
#include <Eigen/Dense>

typedef Eigen::Vector3f V3f;
typedef Eigen::Vector4f V4f;

class EigenHelperFuns
{
public:
  EigenHelperFuns();

  static void gmOrth(V3f &n1,V3f &n2,V3f &n3);

  // This should be templated but wth...
  static Eigen::MatrixXf readMatrix(const std::string &fileName, const int &reservSize=1E6);
  static Eigen::MatrixXd readMatrixd(const std::string &fileName, const int &reservSize=1E6);
  static Eigen::MatrixXd readMatrixdNAN(const std::string &fileName, const int &reservSize=1E6);
  static void readMatrixdBIG(const std::string &fileName, Eigen::MatrixXd &mat);
  static void readMatrixdBIG(const std::string &fileName, Eigen::MatrixXf &mat);
  static Eigen::VectorXf readVec(const std::string &fileName, const int &reservSize=1E6);
  static Eigen::VectorXd readVecd(const std::string &fileName, const int &reservSize=1E6);
  static Eigen::MatrixXd readCSV2Matrix(const std::string &fileName, const int &reservSize=1E6);

  static int writeMat2File(const Eigen::MatrixXf &matrix, const std::string &fileName);
  static int writeMat2Filed(const Eigen::MatrixXd &matrix, const std::string &fileName);

  static void printMatSize(const Eigen::MatrixXf &X, const std::string &MatName="");
  static void printMatSized(const Eigen::MatrixXd &X, const std::string &MatName="");

  static void printEigenVec(const Eigen::VectorXf &vec, const std::string &vecName="");
  static void printMat(const Eigen::MatrixXf &X, const std::string &MatName="");


  static void colwiseMinMaxIdx(const Eigen::MatrixXd &X, Eigen::MatrixXd &idxs,const bool findMaxIdx=true);
  static void rowwiseMinMaxIdx(const Eigen::MatrixXd &X, Eigen::MatrixXd &idxs,const bool findMaxIdx=true);

  static void computeDistMat(const Eigen::MatrixXf &X, Eigen::MatrixXf &D);
  static void removeRow(Eigen::MatrixXf &matrix, const unsigned int &rowToRemove);
  static void removeColumn(Eigen::MatrixXf& matrix, const unsigned int &colToRemove);
  static double euclidSqDist(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2);
  static double chiSqDist(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2);
  static double manhattanDist(const Eigen::VectorXd &vec1, const Eigen::VectorXd &vec2);
  static double histDistKernel(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2, const std::vector<int> &intervals );
  static double histIntersectKernel(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2);
  static Eigen::MatrixXf subSampleMat(const Eigen::MatrixXf &X, const float &percentToKeep);
  static Eigen::VectorXf vector2EigenVec(const std::vector<double> &vec );
  static Eigen::MatrixXd vector2EigenMatrix(const std::vector<std::vector<double> > &vec);
  static void eigenVec2StdVec(const Eigen::VectorXf &eigVec, std::vector<double> &stdVec );
  static void pushEigenVec2StdVec(const Eigen::VectorXf &eigVec, std::vector<double> &stdVec);
  static void rotVec180deg(const V3f &axis, V3f &vec);
  static void rotVecDegAroundAxis(const V3f &axis, const double &deg, V3f& vec);
  static double angleBetweenVectors(const V3f &vec, const V3f &axis, const bool &degrees=false);
  static double signedAngleBetweenVectors(const V3f& vec1, const V3f& vec2, const bool &degrees=false);
  static inline int randIdx(const int &Nmax){ return std::rand() % Nmax; }

  static void zScaling(Eigen::MatrixXd &X);
  static void zScaling(const Eigen::VectorXd &mean, const Eigen::VectorXd &stdev , Eigen::MatrixXd &X);
  static void centerMatrix(Eigen::MatrixXd &X);
  static void centerMatrix(const Eigen::VectorXd &mean, Eigen::MatrixXd &X);
  static void scaleMatrix(const Eigen::VectorXd &stdev, Eigen::MatrixXd &X);
  static void scaleMatrix(Eigen::MatrixXd &X);

  static void cptMean(const Eigen::MatrixXd &X, Eigen::VectorXd &mean);
  static void cptStd(const Eigen::MatrixXd &X, Eigen::VectorXd &stdev);


};


#ifndef DEG2RAD
#define DEG2RAD(x) ((x)*0.017453293)
#endif

#ifndef RAD2DEG
#define RAD2DEG(x) ((x)*57.29578)
#endif

#endif // EIGENHELPERFUNS_H
