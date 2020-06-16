#ifndef PCA_H
#define PCA_H

// std
#include <vector>
#include <Eigen/Dense>

class pca
{

  void centerMatrix(Eigen::MatrixXd &X);
  void scaleMatrix(Eigen::MatrixXd &X);
  void test_scaleMatrix(const Eigen::MatrixXd &X);
  void test_centerMatrix(const Eigen::MatrixXd &X);

  void centerMatrix(Eigen::MatrixXf &X);
  void scaleMatrix(Eigen::MatrixXf &X);
  void test_scaleMatrix(const Eigen::MatrixXf &X);
  void test_centerMatrix(const Eigen::MatrixXf &X);

  int dim_;
  bool scalingOn_;
  /* projMat - V-part of matrix of SVD of X and also projection matrix  */
  Eigen::MatrixXd projMatd;
  Eigen::MatrixXf projMatf;

public:
  pca(const int &dim, const bool &scalingOn);
  pca();

  // Computes the PCA matrix for the given data vector
  // and returns the sub space projected data matrix.
  Eigen::MatrixXd computePCA(const Eigen::MatrixXd &X);
  Eigen::MatrixXf computePCA(const Eigen::MatrixXf &X);

  // Projects a given vector/matrix down on the learned subspace projector
  Eigen::VectorXd computePCAprojection(const Eigen::VectorXd &X);
  Eigen::VectorXf computePCAprojection(const Eigen::VectorXf &X);
  Eigen::MatrixXf computePCAprojection(const Eigen::MatrixXf &X);
  Eigen::MatrixXd computePCAprojection(const Eigen::MatrixXd &X);


  /* Setters & Getters */
  inline void setDim(int dim){dim_ = dim;}
  inline void scalingOn(bool onoff){scalingOn_ = onoff;}
  inline void setProjMat(const Eigen::MatrixXd &pm){projMatd = pm;}
  inline void setProjMat(const Eigen::MatrixXf &pm){projMatf = pm;}
  inline Eigen::MatrixXd getProjMatd(){return projMatd;}
  inline Eigen::MatrixXf getProjMatf(){return projMatf;}

};

#endif // PCA_H
