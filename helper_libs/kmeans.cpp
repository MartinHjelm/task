// std
#include <ctime>
#include <iostream>
#include <stdexcept>
// #include <omp.h>
#if __has_include("omp.h")
# include "omp.h"
#endif

#include <eigenhelperfuns.h>
#include <kmeans.h>
#include <myhelperfuns.h>



kMeans::kMeans(const int &k, const int &maxIter, const int &trials)
    : Ncentroids_(k), maxIter_(maxIter), trials_(trials), intervals_(3, 11) {}

double kMeans::computeK(const Eigen::MatrixXd &X, const int &maxIter,
                        const int &Ntrials) {
  // Nmbr of data points and dimension of data
  int Nvecs = X.rows();
  int Ndims = X.cols();
  // Trial outcome containers
  std::vector<Eigen::MatrixXd> centroidVec(Ntrials);
  std::vector<double> sqrdErrTrials(Ntrials);

  for (int i_trial = 0; i_trial != Ntrials; i_trial++) {

    std::cout << "Starting k-Means trial " << i_trial << std::endl;

    // Centroid assignments
    std::vector<std::vector<int>> r(Ncentroids_);
    std::vector<std::vector<int>> rOld(Ncentroids_);
    // Centroid values
    Eigen::MatrixXd Mu(Ncentroids_, Ndims);

    // Assign random vectors to Mu
    std::srand(std::time(NULL));
    for (int iCentroid = 0; iCentroid != Ncentroids_; iCentroid++) {
      Mu.row(iCentroid) = X.row(MyHelperFuns::randIdx(Nvecs));
    }

    // Start alg.
    for (int iter = 0; iter != maxIter; iter++) {
      // Check for duplicates.
      removeCentroidDoubles(Mu, X);
      printf("Running iteration %d \n", iter);

      rOld = r;
      r.clear();
      r.resize(Ncentroids_, std::vector<int>(0));

      // Find new nearest cluster center for each point
      Eigen::VectorXi NCvec = Eigen::VectorXi::Zero(X.rows());
      #pragma omp parallel for
      for (uint idx = 0; idx < X.rows(); idx++)
      {
        NCvec[idx] = computeNearestCentroidDist(Mu, X.row(idx));
      }

      for (int idx = 0; idx < X.rows(); idx++)
        r[NCvec[idx]].push_back(idx);

      // Check if r hasn't changed then we terminate
      if (iter != 0 && rOld == r) {
        printf("Converged stopping.. \n");
        centroidVec[i_trial] = Mu;
        sqrdErrTrials[i_trial] = computeSqrdError(X, r, Mu);
        printf("Sqrd error: %f\n", sqrdErrTrials[i_trial]);
        break;
      }

      // Check if r has changed for each centroid if not then we dont need to
      // recompute that centroid.
      std::vector<bool> rHasNotChanged(Ncentroids_, false);
      for (int iCentroid = 0; iCentroid != Ncentroids_; iCentroid++) {
        if (r[iCentroid].size() == rOld[iCentroid].size())
          if (r[iCentroid] == rOld[iCentroid])
            rHasNotChanged[iCentroid] = true;
      }

    // Compute new cluster centre for each assignment by taking the mean of every
    // point from
    // given the new assigments.
        for (int iCentroid = 0; iCentroid != Ncentroids_; iCentroid++) {

          // If cluster points hasnt changed so hasnt centroid
          if (rHasNotChanged[iCentroid]) {
            // printf("Centroid %d did not change continuing.. \n",iCentroid);
            continue;
          }
          Eigen::VectorXd newCentroid = Eigen::VectorXd::Zero(X.cols());
          // printf("Running iteration for %d centroid \n",iCentroid);
          // std::cout << "Number of points: " << " " <<  r[iCentroid].size() <<
          // std::endl;

          // newCentroid = Eigen::VectorXd::Zero(X.cols());
          if (r[iCentroid].size() > 0) {
            // For each vector belonging to current center
            std::vector<int>::iterator iter = r[iCentroid].begin();
            for (; iter != r[iCentroid].end(); ++iter) {
              // Eigen::VectorXd vec = X.row(*iter);
              // newCentroid = newCentroid + vec;
              newCentroid += X.row(*iter);
            }
            newCentroid /= r[iCentroid].size();
          } else {
            newCentroid = X.row(MyHelperFuns::randIdx(Nvecs));
          }

          // Assign
          Mu.row(iCentroid) = newCentroid;
        }

    }

    // Ran out of iterations assign to last iteration centroid
    printf("Reached max iterations. Stopping..\n");
    centroidVec[i_trial] = Mu;
    sqrdErrTrials[i_trial] = computeSqrdError(X, r, Mu);
    printf("Sqrd error: %f\n", sqrdErrTrials[i_trial]);
    // MyHelperFuns::print2DVector(r);
    for (int iCentroid = 0; iCentroid != Ncentroids_; iCentroid++)
      std::cout << r[iCentroid].size() << std::endl;
  }
  // End of trials

  int minIdx = MyHelperFuns::minIdxOfVector(sqrdErrTrials);
  centroids = centroidVec[minIdx];
  return sqrdErrTrials[minIdx];
}

double kMeans::computeSqrdError(const Eigen::MatrixXd &X,
                                const std::vector<std::vector<int>> &r,
                                const Eigen::MatrixXd &Mu) {

  double sum = 0.0;

  // For all centroids
  for (int iCentroid = 0; iCentroid != Ncentroids_; iCentroid++) {
    // For all points belonging to the cluster
    for (int iPt = 0; iPt != r[iCentroid].size(); iPt++) {
      // EigenHelperFuns::histDistKernel(Mu.row(iCentroid),X.row(r[iCentroid][iPt]),intervals_);
      sum += EigenHelperFuns::manhattanDist(Mu.row(iCentroid),
                                            X.row(r[iCentroid][iPt]));
    }
  }

  return sum;
}

/* Computes the nearest centroid to a given vector. */
int kMeans::computeNearestCentroidDist(const Eigen::MatrixXd &mu,
                                       const Eigen::VectorXd &vec) {

  // Manhattan Distance(l1) to all mean vectors
  Eigen::VectorXd distVec = (mu.rowwise() - vec.transpose()).cwiseAbs().rowwise().sum();

  // Find index(center) of min distance
  double distMin = distVec.minCoeff();
  int minIdx = 0;

  for (int iRow = 0; iRow != mu.rows(); iRow++)
    if(distVec[iRow] <= distMin)
      minIdx = iRow;

  return minIdx;
}

/* Removes and re-assigns centroids that have converged to the same point. */
void kMeans::removeCentroidDoubles(Eigen::MatrixXd &mu,
                                   const Eigen::MatrixXd &X) {
  std::vector<int> duplicates;
  std::vector<int> intervals(3, 11); // Hard-coded...buuuh
  // Compare indices, if are the same remove them
  for (int iRow = 0; iRow != mu.rows(); iRow++) {
    for (int jRow = iRow + 1; jRow != mu.rows(); jRow++) {
      // double dist =
      // EigenHelperFuns::histDistKernel(mu.row(iRow),mu.row(jRow),intervals);
      double dist = EigenHelperFuns::manhattanDist(mu.row(iRow), mu.row(jRow));

      if (dist < 1E-10) {
        duplicates.push_back(jRow);
        //        MyHelperFuns::printEigenVec(mu.row(iRow),"Vec1");
        //        MyHelperFuns::printEigenVec(mu.row(jRow),"Vec2");
        //        std::cout << "Distance" << dist << std::endl;
      }
    }
  }

  if (duplicates.size() > 0) {
    for (std::vector<int>::iterator iter = duplicates.begin();
         iter != duplicates.end(); ++iter)
      mu.row(*iter) = X.row(MyHelperFuns::randIdx(X.rows()));

    // Do recursive call
    removeCentroidDoubles(mu, X);
  }
}
