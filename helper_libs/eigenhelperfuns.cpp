#include <eigenhelperfuns.h>

// STD
#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <stdexcept>

#include <boost/algorithm/string.hpp>

EigenHelperFuns::EigenHelperFuns() {};


/* Gram-Schmidt Orhtogonalization */
void
EigenHelperFuns::gmOrth(V3f &n1,V3f &n2,V3f &n3)
{
    // 1
    n1.normalize();
    n2 = n2 - n2.dot(n1)*n1;
    n3 = n3 - n3.dot(n1)*n1;
    // 2
    n2.normalize();
    n3 = n3 - n3.dot(n2)*n2;
    // 2
    n3.normalize();
}

double
EigenHelperFuns::signedAngleBetweenVectors(const V3f& vec1, const V3f& vec2, const bool &degrees)
{
    double angle = std::acos(vec1.normalized().dot(vec2.normalized()));
    if(degrees)
        return RAD2DEG(angle);
    else
        return angle;
}

double
EigenHelperFuns::angleBetweenVectors(const V3f& vec1, const V3f& vec2, const bool &degrees)
{
    double angle = std::fabs(std::acos(vec1.normalized().dot(vec2.normalized())));
    angle = (std::min) (angle, M_PI - angle);
    if(degrees)
        return RAD2DEG(angle);
    else
        return angle;
}


void
EigenHelperFuns::rotVec180deg(const V3f &axis, V3f& vec)
{
  // Rotate main axis vector 180deg if it is pointing downwards
    Eigen::AngleAxis<float> rotatePIradAroundSubAxis(M_PI,axis);
    Eigen::Transform<float,3,3> t(rotatePIradAroundSubAxis);
    vec = (t * vec);
}

void
EigenHelperFuns::rotVecDegAroundAxis(const V3f &axis, const double &deg, V3f& vec)
{
    double rad = DEG2RAD(deg);
    Eigen::AngleAxis<float> rotateRadAroundSubAxis(rad,axis);
    Eigen::Transform<float,3,3> t(rotateRadAroundSubAxis);
    vec = t * vec;
}



Eigen::VectorXf
EigenHelperFuns::vector2EigenVec(const std::vector<double> &vec )
{
  Eigen::VectorXf eigVec(vec.size());
  for( int idx=0; idx!=vec.size(); idx++ )
    eigVec(idx) = vec[idx];
  return eigVec;
}


Eigen::MatrixXd
EigenHelperFuns::vector2EigenMatrix(const std::vector<std::vector<double> > &vec)
{
  // This assumes a square 2D vector!
  int rows = vec.size();
  int cols = vec[0].size();

  Eigen::MatrixXd eigMat = Eigen::MatrixXd::Zero(rows,cols);
  for( int idx=0; idx!=rows; idx++ )
  {
    assert(vec[idx].size()==cols || !(std::cerr << "Vector needs to be square!!!" << std::endl ) );
    for( int jdx=0; jdx!=cols; jdx++ )
    {
      eigMat(idx,jdx) = vec[idx][jdx];
    }
  }
return eigMat;
}



void
EigenHelperFuns::eigenVec2StdVec(const Eigen::VectorXf &eigVec, std::vector<double> &stdVec )
{
  stdVec.clear();
  stdVec.resize(eigVec.size());
  for(int idx=0; idx!=stdVec.size(); idx++)
    stdVec[idx] = eigVec(idx);
}


void
EigenHelperFuns::pushEigenVec2StdVec(const Eigen::VectorXf &eigVec, std::vector<double> &stdVec)
{
    for(int i=0; i!=eigVec.size(); i++)
        stdVec.push_back(eigVec(i));
}


void EigenHelperFuns::removeRow(Eigen::MatrixXf &matrix, const unsigned int &rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

void EigenHelperFuns::removeColumn(Eigen::MatrixXf &matrix, const unsigned int &colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}



void
EigenHelperFuns::rowwiseMinMaxIdx(const Eigen::MatrixXd &X, Eigen::MatrixXd &idxs,const bool findMaxIdx)
{
    Eigen::MatrixXf::Index rowIdx;
    for(int iRow = 0; iRow < X.rows(); iRow++)
    {
      findMaxIdx ? X.row(iRow).maxCoeff(&rowIdx) : X.row(iRow).minCoeff(&rowIdx);
      idxs(iRow,0) = rowIdx;
    }
}

void
EigenHelperFuns::colwiseMinMaxIdx(const Eigen::MatrixXd &X, Eigen::MatrixXd &idxs,const bool findMaxIdx)
{
    Eigen::MatrixXf::Index colIdx;
    for(int iCol = 0; iCol < X.cols(); iCol++)
    {
      findMaxIdx ? X.col(iCol).maxCoeff(&colIdx) : X.col(iCol).minCoeff(&colIdx);
      idxs(0,iCol) = colIdx;
    }
}




/************ READING AND WRITING MAT FILES *****************/

int
EigenHelperFuns::writeMat2File(const Eigen::MatrixXf &matrix, const std::string &fileName)
{
  std::ofstream out( fileName.c_str() );

  if (out.is_open())
    out << matrix;
  else
    return 0;

  out.close();
  return 1;
}

int
EigenHelperFuns::writeMat2Filed(const Eigen::MatrixXd &matrix, const std::string &fileName)
{
  std::ofstream out( fileName.c_str() );


// Eigen::IOFormat  csvFormat( Eigen::StreamPrecision,
//                 0,
//                 ",");
  if (out.is_open())
    out << matrix;
  else
    return 0;

  out.close();
  return 1;
}



Eigen::MatrixXf
EigenHelperFuns::readMatrix(const std::string &fileName, const int &reservSize)
{
  int cols = 0, rows = 0;
  std::vector<double> buff;
  buff.reserve(reservSize);

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  while (! infile.eof())
  {
      // For each row in file
      std::string line;
      std::getline(infile, line);

      int temp_cols = 0;
      std::stringstream stream(line);
      while(! stream.eof())
      {
        double number;
        stream >> number;
        temp_cols++;
        buff.push_back( number);
    }
    if (temp_cols == 0)
        continue;

    if (cols == 0)
        cols = temp_cols;

    rows++;
  }

  infile.close();

  //  std::cout << "Finished reading file converting to mat" << std::endl;

  //  rows--;
  //std::cout << buff.size() << " "  << rows << " " << cols <<  std::endl;
    // Populate matrix with numbers.
  Eigen::MatrixXf result(rows,cols);
  for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        result(i,j) = buff[ cols*i+j ];

  //  printMatSize(result);

  return result;
}

Eigen::MatrixXd
EigenHelperFuns::readMatrixd(const std::string &fileName, const int &reservSize)
{
  int cols = 0, rows = 0;
  std::vector<double> buff;
  buff.reserve(reservSize);

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

while (! infile.eof())
{
    // For each row in file
    std::string line;
    std::getline(infile, line);

    int temp_cols = 0;
    std::stringstream stream(line);
    while(! stream.eof())
    {
      double number;
      stream >> number;
      temp_cols++;
      buff.push_back( number);
  }
  if (temp_cols == 0)
      continue;

  if (cols == 0)
      cols = temp_cols;

  rows++;
}

infile.close();

//  std::cout << "Finished reading file converting to mat" << std::endl;

//  rows--;
//std::cout << buff.size() << " "  << rows << " " << cols <<  std::endl;
  // Populate matrix with numbers.
Eigen::MatrixXd result(rows,cols);
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      result(i,j) = buff[ cols*i+j ];

  return result;
}



Eigen::MatrixXd
EigenHelperFuns::readMatrixdNAN(const std::string &fileName, const int &reservSize)
{
  int cols = 0, rows = 0;
  std::vector<double> buff;
  buff.reserve(reservSize);

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);

    bool isLineNan = false;

    int temp_cols = 0;
    std::stringstream stream(line);
    while(! stream.eof())
    {
      double number;
      stream >> number;
      if(std::isnan(number))
      {
        printf("Lines is nan!");
        isLineNan = true;
        break;
      }
      temp_cols++;
      buff.push_back( number);
    }
  if (isLineNan)
  {
    continue;
  }

  if (temp_cols == 0)
      continue;

  if (cols == 0)
      cols = temp_cols;

  rows++;
}

infile.close();

// Populate matrix with numbers.
Eigen::MatrixXd result(rows,cols);
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      result(i,j) = buff[ cols*i+j ];

  return result;
}



Eigen::MatrixXd
EigenHelperFuns::readCSV2Matrix(const std::string &fileName, const int &reservSize)
{
  int cols = 0, rows = 0;
  std::vector<double> buff;
  buff.reserve(reservSize);

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);

    std::vector <std::string> fields;
    boost::split( fields, line, boost::is_any_of( " ," ), boost::token_compress_on );

    for(std::vector<std::string>::iterator it = fields.begin(); it != fields.end(); it++ )
    {
      buff.push_back(std::stod(*it));
    }

    if (fields.size() != 0)
      cols = fields.size();

    rows++;
  }


infile.close();

// Populate matrix with numbers.
Eigen::MatrixXd result(rows,cols);
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      result(i,j) = buff[ cols*i+j ];

  return result;
}


void
EigenHelperFuns::readMatrixdBIG(const std::string &fileName, Eigen::MatrixXd &mat)
{
  int Nrows = 0;
  int Ncols = 0;
  // Count the number of rows and columns
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);

    // Count the number of columns in the first row or any other row not containing NaN
    if( Nrows==0 || Ncols==0 )
    {
      int temp_cols = 0;
      std::stringstream stream(line);
      while(! stream.eof())
      {
        double number;
        stream >> number;
        if(std::isnan(number))
        {
          printf("Lines is nan!");
          Ncols=0;
          break;
        }
        Ncols++;
      }
    }
    Nrows++;
  }
  infile.close();

  std::cout << "Reading Matrix" << std::endl << std::flush;

  // Allocate Eigen Mat
  mat = Eigen::MatrixXd::Zero(Nrows,Ncols);

  // Read the file again copying in values
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  int iRow = 0;
  int iCol = 0;
  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);
    bool isLineNan = false;

    iCol = 0;
    std::stringstream stream(line);
    while(! stream.eof())
    {
      double number;
      stream >> number;
      if(std::isnan(number))
      {
        printf("Lines is nan!");
        break;
      }

      if(iRow<Nrows && iCol<Ncols){ mat(iRow,iCol) = number; }
      else{std::cout << "Tried writing to " << iRow << "," << iCol << " but " << Nrows << "," << Ncols  << std::endl << std::flush;}

      iCol++;
    }
    iRow++;
  }

  infile.close();
    std::cout << "Done..." << std::endl << std::flush;
}


void
EigenHelperFuns::readMatrixdBIG(const std::string &fileName, Eigen::MatrixXf &mat)
{
  int Nrows = 0;
  int Ncols = 0;
  // Count the number of rows and columns
  std::ifstream infile;
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);

    // Count the number of columns in the first row or any other row not containing NaN
    if( Nrows==0 || Ncols==0 )
    {
      int temp_cols = 0;
      std::stringstream stream(line);
      while(! stream.eof())
      {
        double number;
        stream >> number;
        if(std::isnan(number))
        {
          printf("Lines is nan!");
          Ncols=0;
          break;
        }
        Ncols++;
      }
    }
    Nrows++;
  }
  infile.close();

  std::cout << "Reading Matrix" << std::endl << std::flush;

  // Allocate Eigen Mat
  mat = Eigen::MatrixXf::Zero(Nrows,Ncols);

  // Read the file again copying in values
  infile.open(fileName.c_str());
  if(!infile.is_open())
    throw std::runtime_error("File "+fileName+" not found or something else went wrong when reading..");

  int iRow = 0;
  int iCol = 0;
  while (! infile.eof())
  {
    // For each row in file
    std::string line;
    std::getline(infile, line);
    bool isLineNan = false;

    iCol = 0;
    std::stringstream stream(line);
    while(! stream.eof())
    {
      double number;
      stream >> number;
      if(std::isnan(number))
      {
        printf("Lines is nan!");
        break;
      }

      if(iRow<Nrows && iCol<Ncols){mat(iRow,iCol) = number;}
      else{std::cout << "Tried writing to " << iRow << "," << iCol << " but " << Nrows << "," << Ncols  << std::endl << std::flush;}

      iCol++;
    }
    iRow++;
  }

  infile.close();
    std::cout << "Done..." << std::endl << std::flush;
}



Eigen::VectorXf
EigenHelperFuns::readVec(const std::string &fileName, const int &reservSize)
{
  Eigen::MatrixXf X = EigenHelperFuns::readMatrix(fileName);
  // Transpose in place to get row, row, ...
  X.transposeInPlace();
  // Store matrix in vector using maps
  return Eigen::VectorXf(Eigen::Map<Eigen::VectorXf>(X.data(), X.cols()*X.rows()));
}


Eigen::VectorXd
EigenHelperFuns::readVecd(const std::string &fileName, const int &reservSize)
{
  Eigen::MatrixXd X = EigenHelperFuns::readMatrixd(fileName);
  // Transpose in place to get row, row, ...
  X.transposeInPlace();
  // Store matrix in vector using maps
  return Eigen::VectorXd(Eigen::Map<Eigen::VectorXd>(X.data(), X.cols()*X.rows()));
}


double
EigenHelperFuns::chiSqDist(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2)
{
  // sum( (xi-yi)^2 / (xi+yi) ) / 2;
  double dist = 0;
  int Nelems = vec1.size();

  Eigen::VectorXf vecDiffSq = (vec1 - vec2).array().square();
//  vecDiffSq = vecDiffSq.array().square();
  Eigen::VectorXf vecSum = vec1 + vec2;

  for(int iIdx=0; iIdx!=Nelems; iIdx++)
  {
    if(vecSum(iIdx)>1E-6)
      dist += vecDiffSq(iIdx) / vecSum(iIdx);
}

dist *= 0.5;

return dist;
}

double
EigenHelperFuns::euclidSqDist(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2)
{  return (vec1 - vec2).array().square().sum(); }



double
EigenHelperFuns::histDistKernel(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2, const std::vector<int> &intervalSize )
{
  assert(vec1.size()==vec2.size());

  double dist = 0;
  int idx = 0;

  for(int iVal=0; iVal!=intervalSize.size(); iVal++)
  {
     dist += chiSqDist(vec1.segment(idx,intervalSize[iVal]), vec2.segment(idx,intervalSize[iVal]));
     idx += intervalSize[iVal];
 }
 return dist;
}


double
EigenHelperFuns::histIntersectKernel(const Eigen::VectorXf &vec1, const Eigen::VectorXf &vec2)
{
  assert(vec1.size()==vec2.size());

  double dist = 0.0;
  for(int idx=0; idx!=vec1.size(); idx++)
    dist += std::min(vec1(idx),vec2(idx));

  return dist;
}


double
EigenHelperFuns::manhattanDist(const Eigen::VectorXd &vec1, const Eigen::VectorXd &vec2)
{
  assert(vec1.size()==vec2.size());
  return (vec1-vec2).cwiseAbs().sum();
}






void
EigenHelperFuns::printMatSize(const Eigen::MatrixXf &X, const std::string &MatName)
{
  std::cout << MatName << " " << X.rows() << " x " << X.cols() << std::endl;
}

void
EigenHelperFuns::printMatSized(const Eigen::MatrixXd &X, const std::string &MatName)
{
  std::cout << MatName << " " << X.rows() << " x " << X.cols() << std::endl;
}

void
EigenHelperFuns::printMat(const Eigen::MatrixXf &X, const std::string &MatName)
{
  std::cout << MatName << std::endl << X << std::endl;
}



void
EigenHelperFuns::printEigenVec(const Eigen::VectorXf &vec, const std::string &vecName)
{
  using namespace std;
  if(vecName.size()!=0)
    cout << vecName << " (";
      else
        cout << "(";
          for(int idx=0; idx!=vec.size()-1; idx++) { cout << vec(idx) << ", "; }
              cout << vec(vec.size()-1) << ")" << endl;
}





Eigen::MatrixXf
EigenHelperFuns::subSampleMat(const Eigen::MatrixXf &X, const float &percentToKeep )
{
  assert(percentToKeep<1 && percentToKeep>0);

  // Get the indices to keep
  int Ndata = X.rows();
  int Nnew = Ndata * percentToKeep;

  // Allocate Matrix
  Eigen::MatrixXf Xnew(Nnew,X.cols());

  std::srand(std::time(NULL)); // Reset rnd generator
  //#pragma omp parallel for
  for(int iRow=0; iRow<Nnew; iRow++)
  {
    Xnew.row(iRow) = X.row(randIdx(Ndata));
}

return Xnew;
}


void
EigenHelperFuns::computeDistMat(const Eigen::MatrixXf &X, Eigen::MatrixXf &D)
{
    const int N = X.rows();

    // Allocate parts of the expression
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> XX, YY, XY;
    XX.resize(N,1);
    YY.resize(1,N);
    XY.resize(N,N);
    D.resize(N,N);
    // Compute norms
    XX = X.array().square().rowwise().sum();
    YY = X.array().square().rowwise().sum().transpose();
    XY = (2*X)*X.transpose();

    // Compute final expression
    D = XX * Eigen::MatrixXf::Ones(1,N);
    D = D + Eigen::MatrixXf::Ones(N,1) * YY;
    D = D - XY;
}



void
EigenHelperFuns::zScaling(Eigen::MatrixXd &X)
{
  EigenHelperFuns::centerMatrix(X);
  EigenHelperFuns::scaleMatrix(X);
}

void
EigenHelperFuns::zScaling(const Eigen::VectorXd &mean, const Eigen::VectorXd &stdev , Eigen::MatrixXd &X)
{
  EigenHelperFuns::centerMatrix(mean, X);
  EigenHelperFuns::scaleMatrix(stdev, X);
}




/* Removes the mean of the row vectors from each row. */
void
EigenHelperFuns::centerMatrix(Eigen::MatrixXd &X)
{
  Eigen::VectorXd mean = X.colwise().mean();
  EigenHelperFuns::centerMatrix(mean, X);
}


void
EigenHelperFuns::centerMatrix(const Eigen::VectorXd &mean, Eigen::MatrixXd &X)
{
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    X.col(iCol).array() -= mean(iCol);
    // Use the fast access pointer iterator in eigen!
//    for (int i = 0; i < X.size(); i++)
//      *(X.data() + X.cols()*iCol + i) -= m;
  }

}

void
EigenHelperFuns::scaleMatrix(const Eigen::VectorXd &stdev, Eigen::MatrixXd &X)
{

  // Number of datapoints minus one
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    if(stdev(iCol)>1E-16)
      X.col(iCol).array() /= stdev(iCol);

    // Use the fast access pointer iterator in eigen!
//    for (int i = 0; i < X.size(); i++)
//      *(X.data() + X.cols()*iCol + i) /= divisor;
  }

}


/* Scale Matrix by unbiased std deviation so each dimension sums to one. */
void
EigenHelperFuns::scaleMatrix(Eigen::MatrixXd &X)
{
  Eigen::VectorXd stdev;
  EigenHelperFuns::cptStd(X,stdev);
  EigenHelperFuns::scaleMatrix(stdev, X);
}

void
EigenHelperFuns::cptMean(const Eigen::MatrixXd &X, Eigen::VectorXd &mean)
{
  mean = X.colwise().mean();
}

void
EigenHelperFuns::cptStd(const Eigen::MatrixXd &X, Eigen::VectorXd &stdev)
{
  stdev =  Eigen::VectorXd::Zero(X.cols());
  double factor = X.rows()-1;
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    Eigen::VectorXd colVec = X.col(iCol);
    stdev(iCol) = std::sqrt( colVec.dot(colVec) / factor );
  }
}
