#ifndef KMEDOIDS_H
#define KMEDOIDS_H

// std
#include <vector>
#include <Eigen/Dense>


class kMedoids
{
  int Ncentroids_;
  int maxIter_;
  int trials_;
  std::vector<int> intervals_;

  int computeNearestCentroidDist(const Eigen::MatrixXd &mu, const Eigen::VectorXd &vec);
  void removeCentroidDoubles(Eigen::MatrixXd &mu,const Eigen::MatrixXd &X );
  double computeSqrdError(const Eigen::MatrixXd &X, const std::vector<std::vector<int> > &r, const Eigen::MatrixXd &Mu);

public:

  Eigen::MatrixXd centroids;

  kMedoids(const int &k=5, const int &maxIter=100, const int &trials=1);
  double computeK(const Eigen::MatrixXd &X, const int &maxIter, const int &Ntrials=1);

  inline void setNCentroids(int k){Ncentroids_=k;}
  inline void setmaxIter_(int n){maxIter_=n;}
};

#endif // KMEDOIDS_H
