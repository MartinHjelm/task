#include "pca.h"

// std
#include <iostream>
#include <stdexcept>

#include <Eigen/Eigenvalues>


pca::pca(const int &dim, const bool &scalingOn) :
  dim_(dim),
  scalingOn_(scalingOn)
{}

pca::pca() : dim_(1), scalingOn_(false)
{}


Eigen::MatrixXd
pca::computePCA(const Eigen::MatrixXd &data)
{

    Eigen::MatrixXd X = data;

  /*
  The covariance
  X'*X is hermitian... so
    X'*X = W*D*W'
  where W is the eigen vectors and D the diagonal eigen value matrix.
  We have svd(X) = USV' so
    X'*X =  (USV')' * USV' = VS'U' * USV' = {U*U' = I} = V(S'S)V'
  so
    D = S*S' and W = V;
  %}

  % 1. Remove mean
  X = X - ones(M,1) * mean(X,1);

  % 1.5 If scaling is on, scale each dimensions variance to sum to one.
  if scaleOn
    X = X ./ (ones(M,1)*std(X,[],1));
  end

  % 2. Compute the SVD (we get already sorted eigen values and eigen vectors!)
  % D - eigenvalues and V eigen vectors
  [~,S,V] = svd( X / sqrt(N-1) );
  % [U,S,V] = svd ( X / sqrt(N-1), 'econ');

  % 3. Choose the m biggest
  V = V(:,1:m);
  S = diag(S'*S);
  D = S(1:m);

  % 4. Calculate the projection of X on to the PC m-subspace in the biggest variance direction.
  Z = X * V;
  */

  // Make matrix zero mean
  centerMatrix(X);
//  test_centerMatrix();

  // If scaling is on scale each dimension to sum to one.
  if(scalingOn_)
  {
    scaleMatrix(X);
//    test_scaleMatrix();
  }

  // Check if D > N
  if (X.cols() > X.rows())
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(X.rows()*X.rows());
    es.compute(X.transpose()*X);
    // std::cout << "The eigenvalues of A are:" << std::endl << es.eigenvalues() << std::endl;
    // std::cout << "The matrix of eigenvectors, V, is:" << std::endl << es.eigenvectors() << std::endl << std::endl;
    projMatd = Eigen::MatrixXd::Zero(X.cols(),dim_);
    for(int i = 0; i != dim_; i++)
    {
      projMatd.col(i).array() = (1./std::sqrt(es.eigenvalues()(i)*X.rows())) *
                        X.transpose()*es.eigenvectors().col(i);
    }
    return data * projMatd;
  }
  else
  {
    // Divide matrix by sqrt dim -1 so SVD trick will work.
    X /= std::sqrt(X.cols()-1);

    // Do Eigen SVD.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinV);
  //  std::cout << "Matrix U " << std::endl <<  svd.matrixU() << std::endl  << std::endl;
  //  std::cout << "SingVals S " << std::endl << svd.singularValues() << std::endl  << std::endl;               ;
  //  std::cout << "Matrix V " << std::endl << svd.matrixV() << std::endl  << std::endl;
  //  std::cout << "Product U*V " << std::endl << svd.matrixU() * (svd.matrixV()).transpose() << std::endl  << std::endl;

  //  U = svd.matrixU();
  //  S = svd.singularValues();
    projMatd = (svd.matrixV()).block(0,0,(svd.matrixV()).rows(),dim_);
  //  std::cout << projMat.rows() << " " << projMat.cols() << std::endl;
  //  std::cout << (svd.matrixV()).rows() << " " << (svd.matrixV()).cols() << std::endl;
  //    std::cout << data.rows() << " " << data.cols() << std::endl;
    // Return projection of original data.
  //  Eigen::MatrixXd Z = data * projMat;
    return data * projMatd;
  }
}

Eigen::MatrixXf
pca::computePCA(const Eigen::MatrixXf &data)
{

    Eigen::MatrixXf X = data;

  /*
  The covariance
  X'*X is hermitian... so
    X'*X = W*D*W'
  where W is the eigen vectors and D the diagonal eigen value matrix.
  We have svd(X) = USV' so
    X'*X =  (USV')' * USV' = VS'U' * USV' = {U*U' = I} = V(S'S)V'
  so
    D = S*S' and W = V;
  %}

  % 1. Remove mean
  X = X - ones(M,1) * mean(X,1);

  % 1.5 If scaling is on, scale each dimensions variance to sum to one.
  if scaleOn
    X = X ./ (ones(M,1)*std(X,[],1));
  end

  % 2. Compute the SVD (we get already sorted eigen values and eigen vectors!)
  % D - eigenvalues and V eigen vectors
  [~,S,V] = svd( X / sqrt(N-1) );
  % [U,S,V] = svd ( X / sqrt(N-1), 'econ');

  % 3. Choose the m biggest
  V = V(:,1:m);
  S = diag(S'*S);
  D = S(1:m);

  % 4. Calculate the projection of X on to the PC m-subspace in the biggest variance direction.
  Z = X * V;
  */

  // Make matrix zero mean
  centerMatrix(X);
//  test_centerMatrix();

  // If scaling is on scale each dimension to sum to one.
  if(scalingOn_)
  {
    scaleMatrix(X);
//    test_scaleMatrix();
  }

  // Check if D > N
  if (X.cols() > X.rows())
  {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(X.rows()*X.rows());
    es.compute(X.transpose()*X);
    // std::cout << "The eigenvalues of A are:" << std::endl << es.eigenvalues() << std::endl;
    // std::cout << "The matrix of eigenvectors, V, is:" << std::endl << es.eigenvectors() << std::endl << std::endl;
    projMatf = Eigen::MatrixXf::Zero(X.cols(),dim_);
    for(int i = 0; i != dim_; i++)
    {
      projMatf.col(i).array() = (1./std::sqrt(es.eigenvalues()(i)*X.rows())) *
                        X.transpose()*es.eigenvectors().col(i);
    }
    return data * projMatf;
  }
  else
  {
    // Divide matrix by sqrt dim -1 so SVD trick will work.
    X /= std::sqrt(X.cols()-1);

    // Do Eigen SVD.
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);
  //  std::cout << "Matrix U " << std::endl <<  svd.matrixU() << std::endl  << std::endl;
  //  std::cout << "SingVals S " << std::endl << svd.singularValues() << std::endl  << std::endl;               ;
  //  std::cout << "Matrix V " << std::endl << svd.matrixV() << std::endl  << std::endl;
  //  std::cout << "Product U*V " << std::endl << svd.matrixU() * (svd.matrixV()).transpose() << std::endl  << std::endl;

  //  U = svd.matrixU();
  //  S = svd.singularValues();
    projMatf = (svd.matrixV()).block(0,0,(svd.matrixV()).rows(),dim_);
  //  std::cout << projMat.rows() << " " << projMat.cols() << std::endl;
  //  std::cout << (svd.matrixV()).rows() << " " << (svd.matrixV()).cols() << std::endl;
  //    std::cout << data.rows() << " " << data.cols() << std::endl;
    // Return projection of original data.
  //  Eigen::MatrixXd Z = data * projMat;
    return data * projMatf;
  }
}

/* Removes the mean of the row vectors from each row. */
void
pca::centerMatrix(Eigen::MatrixXd &X)
{
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    float m = (X.col(iCol)).mean();
    X.col(iCol).array() -= m;
    // Use the fast access pointer iterator in eigen!
//    for (int i = 0; i < X.size(); i++)
//      *(X.data() + X.cols()*iCol + i) -= m;
  }

}

void
pca::centerMatrix(Eigen::MatrixXf &X)
{
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    float m = (X.col(iCol)).mean();
    X.col(iCol).array() -= m;
    // Use the fast access pointer iterator in eigen!
//    for (int i = 0; i < X.size(); i++)
//      *(X.data() + X.cols()*iCol + i) -= m;
  }

}

/* Scale Matrix by unbiased std deviation so each dimension sums to one. */
void
pca::scaleMatrix(Eigen::MatrixXd &X)
{

  // Number of datapoints minus one
  double factor = X.rows()-1;
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    Eigen::VectorXd colVec = X.col(iCol);
//    float divisor = std::sqrt( colVec.dot(colVec) / factor );
    X.col(iCol) /= std::sqrt( colVec.dot(colVec) / factor );

    // Use the fast access pointer iterator in eigen!
//    for (int i = 0; i < X.size(); i++)
//      *(X.data() + X.cols()*iCol + i) /= divisor;
  }

}

void
pca::scaleMatrix(Eigen::MatrixXf &X)
{

  // Number of datapoints minus one
  double factor = X.rows()-1;
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    Eigen::VectorXf colVec = X.col(iCol);
//    float divisor = std::sqrt( colVec.dot(colVec) / factor );
    X.col(iCol) /= std::sqrt( colVec.dot(colVec) / factor );

    // Use the fast access pointer iterator in eigen!
//    for (int i = 0; i < X.size(); i++)
//      *(X.data() + X.cols()*iCol + i) /= divisor;
  }

}


Eigen::VectorXd
pca::computePCAprojection(const Eigen::VectorXd &X)
{
  if(X.rows()!=projMatd.rows())
  {
    std::cout << X.rows()  << "x" << X.cols() << " " << projMatd.rows() << "x" << projMatd.cols() << std::endl;
    throw std::runtime_error("Dimensions mismatch!!\n");
  }
  return X.transpose() * projMatd;
}

Eigen::VectorXf
pca::computePCAprojection(const Eigen::VectorXf &X)
{
  if(X.rows()!=projMatf.rows())
  {
    std::cout << X.rows()  << "x" << X.cols() << " " << projMatf.rows() << "x" << projMatf.cols() << std::endl;
    throw std::runtime_error("Dimensions mismatch!!\n");
  }
  return X.transpose() * projMatf;
}


Eigen::MatrixXd
pca::computePCAprojection(const Eigen::MatrixXd &X)
{ return X * projMatd; }

Eigen::MatrixXf
pca::computePCAprojection(const Eigen::MatrixXf &X)
{ return X * projMatf; }


void
pca::test_centerMatrix(const Eigen::MatrixXd &X)
{
  int colsOk = 0;
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    float m = (X.col(iCol)).mean();
    if(std::fabs(m)<1E-6)
      colsOk++;
  }
  if(colsOk==X.cols())
    std::cout << "Centering OK!" << std::endl;
  else
    std::cout << "Centering not OK!" << std::endl;
}

void
pca::test_centerMatrix(const Eigen::MatrixXf &X)
{
  int colsOk = 0;
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    float m = (X.col(iCol)).mean();
    if(std::fabs(m)<1E-6)
      colsOk++;
  }
  if(colsOk==X.cols())
    std::cout << "Centering OK!" << std::endl;
  else
    std::cout << "Centering not OK!" << std::endl;
}


void
pca::test_scaleMatrix(const Eigen::MatrixXd &X)
{
  int colsOk = 0;
  double factor = X.rows()-1;
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    Eigen::VectorXd colVec = X.col(iCol);
    float m = colVec.mean();
    colVec.array() -= m;
    float colStd = std::sqrt( colVec.dot(colVec) / factor );

    if(std::fabs(colStd-1.0)<1E-6)
      colsOk++;
  }

  if(colsOk==X.cols())
    std::cout << "Scaling OK!" << std::endl;
  else
    std::cout << "Scaling not OK!" << std::endl;

}

void
pca::test_scaleMatrix(const Eigen::MatrixXf &X)
{
  int colsOk = 0;
  double factor = X.rows()-1;
  // For each col, i.e., dimension.
  for(int iCol=0; iCol!=X.cols(); iCol++)
  {
    Eigen::VectorXf colVec = X.col(iCol);
    float m = colVec.mean();
    colVec.array() -= m;
    float colStd = std::sqrt( colVec.dot(colVec) / factor );

    if(std::fabs(colStd-1.0)<1E-6)
      colsOk++;
  }

  if(colsOk==X.cols())
    std::cout << "Scaling OK!" << std::endl;
  else
    std::cout << "Scaling not OK!" << std::endl;

}
