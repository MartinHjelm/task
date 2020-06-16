#include <kmedoids.h>

// std
#include <iostream>
#include <stdexcept>
#include <ctime>

// Boost
#include <boost/unordered_map.hpp>

#include <myhelperfuns.h>
#include <eigenhelperfuns.h>


kMedoids::kMedoids(const int &k, const int &maxIter, const int &trials) :
    Ncentroids_(k),
    maxIter_(maxIter),
    trials_(trials),
    intervals_(3,11)
{}



double
kMedoids::computeK(const Eigen::MatrixXd &X, const int &maxIter, const int &Ntrials )
{
    // Nmbr of data points and dimension of data
    int Nvecs = X.rows();
    int Ndims = X.cols();
    // Trial outcome containers
    std::vector<Eigen::MatrixXd> centroidVec(Ntrials);
    std::vector<double> sqrdErrTrials(Ntrials);
    // Cache inter point distances in hash table
    boost::unordered_map<int,double> distmap;

    for(int i_trial=0; i_trial!=Ntrials; i_trial++)
    {

        std::cout << "Starting k-Medoids trial " << i_trial << std::endl;
        // EigenHelperFuns::printMatSize(X,"X");

        // Centroid values
        Eigen::MatrixXd Mu(Ncentroids_,Ndims);
        // Centroid assignment
        std::vector<std::vector<int> > r(Ncentroids_);
        std::vector<std::vector<int> > rOld(Ncentroids_);

        // Assign random vectors to Mu
        std::srand(std::time(NULL));
        for(int iCentroid=0; iCentroid!=Ncentroids_; iCentroid++)
        { Mu.row(iCentroid) = X.row(MyHelperFuns::randIdx(Nvecs)); }


        for(int iter=0; iter!=maxIter_; iter++)
        {
            // Check for duplicates.
            removeCentroidDoubles(Mu,X);

            printf("Running iteration %d \n",iter);
            rOld = r;
            r.clear();
            r.resize(Ncentroids_,std::vector<int>(0));

            // Find new nearest cluster center for each point
            for(int iVec=0; iVec!=Nvecs; iVec++)
            {
                Eigen::VectorXd vec = X.row(iVec);
                r[computeNearestCentroidDist(Mu,vec)].push_back(iVec);
            }

            // Check if r hasnt changed then we terminate
            if(iter!=0 && rOld == r)
            {
                printf("Converged stopping.. \n");
                centroidVec[i_trial] = Mu;
                sqrdErrTrials[i_trial] = computeSqrdError(X,r,Mu);
                break;
            }

            // Check if clusters have changed if not then we dont need to recompute that centroid.
            std::vector<bool> rHasChanged(Ncentroids_,true);
            for(int iCentroid=0; iCentroid!=Ncentroids_; iCentroid++)
            {
                if( r[iCentroid].size() == rOld[iCentroid].size() )
                    if(r[iCentroid] == rOld[iCentroid])
                        rHasChanged[iCentroid] = false;
            }


            // Compute new cluster centre for each assignment by finding the medoids,
            // that is, the point that are closests to each of the points of the cluster.
            for(int iCentroid=0; iCentroid!=Ncentroids_; iCentroid++)
            {
                printf("Running iteration for %d centroid \n",iCentroid);
                if(!rHasChanged[iCentroid])
                    continue;

                int N_points = r[iCentroid].size();
                if(N_points)
                {

                std::cout << "Number of points: " << " " <<  N_points << std::endl;

                // For each vector belonging to the current center compute summed distance to
                // each other point
                std::vector<double> distSums(N_points,0.0);

                for(int ii=0; ii!=N_points; ++ii)
                {
                    for(int jj=0; jj!=N_points; ++jj)
                    {
                        if(ii==jj) continue;

                        //std::cout << "1" << std::endl;
                        int iIdx = r[iCentroid][ii];
                        int jIdx = r[iCentroid][jj];

                        //std::cout << "2" << std::endl;
                        // Check if key exists
                        int key;
                        if(iIdx<jIdx) { key = Nvecs * iIdx + jIdx; }
                        else { key = Nvecs * jIdx + iIdx; }

                        // If key dist does not exist compute it
                        if(distmap.find(key) == distmap.end())
                        {
                            double d = 0;
                            if( (X.row(iIdx)-X.row(jIdx)).array().abs().sum() > 1E-6 )
                                //d = EigenHelperFuns::histDistKernel(X.row(iIdx),X.row(jIdx),intervals_);
                                d = EigenHelperFuns::manhattanDist(X.row(iIdx),X.row(jIdx));
                            //distmap.emplace(key,d);
                            distmap[key] = d;
                            distSums[ii] += d;
                            //std::cout << key << " " << distmap.at(key) << " " << d << std::endl;
                        }
                        else
                        {
                            distSums[ii] += distmap.at(key);
                        }
                        //std::cout << "3" << std::endl;

                    }
                }

                int minIdx = MyHelperFuns::minIdxOfVector(distSums);
                Mu.row(iCentroid) = X.row(r[iCentroid][minIdx]);
            }
            }
        }

        // Ran out of iterations assign to last iteration centroid
        printf("Reached max iterations. Stopping..\n");
        centroidVec[i_trial] = Mu;
        sqrdErrTrials[i_trial] = computeSqrdError(X,r,Mu);
    }
    // End of trials

    int minIdx = MyHelperFuns::minIdxOfVector(sqrdErrTrials);
    centroids = centroidVec[minIdx];
    return sqrdErrTrials[minIdx];

}



double
kMedoids::computeSqrdError(const Eigen::MatrixXd &X, const std::vector<std::vector<int> > &r, const Eigen::MatrixXd &Mu)
{

    double sum = 0.0;

    // For all centroids
    for(int iCentroid=0; iCentroid!=Ncentroids_; iCentroid++)
    {
        // For all points belonging to the cluster
        for(int iPt=0; iPt!=r[iCentroid].size(); iPt++)
        {
         //   sum += EigenHelperFuns::histDistKernel(Mu.row(iCentroid),X.row(r[iCentroid][iPt]),intervals_);
            sum += EigenHelperFuns::manhattanDist(Mu.row(iCentroid),X.row(r[iCentroid][iPt]));
        }
    }

    return sum;
}



/* Computes the nearest centroid to a given vector. */
int
kMedoids::computeNearestCentroidDist (const Eigen::MatrixXd &mu, const Eigen::VectorXd &vec )
{
    std::vector<double> distances(mu.rows(),INFINITY);
    //#pragma omp parallel for shared (output) private (nn_indices, nn_dists) num_threads(threads_)
    for(int iRow=0; iRow!=mu.rows(); iRow++)
    {
        //distances[iRow] = EigenHelperFuns::histDistKernel(mu.row(iRow),vec,intervals_);
        distances[iRow] = EigenHelperFuns::manhattanDist(mu.row(iRow),vec);
    }

    int idx = MyHelperFuns::minIdxOfVector(distances);

    return idx;
}



/* Removes and re-assigns centroids that have converged to the same point. */
void
kMedoids::removeCentroidDoubles (Eigen::MatrixXd &mu,const Eigen::MatrixXd &X )
{
    std::vector<int> duplicates;
    std::vector<int> intervals(3,11); // Hard-coded...buuuh
    // Compare indices, if are the same remove them
    for(int iRow=0; iRow!=mu.rows(); iRow++)
    {
        for(int jRow=iRow+1; jRow!=mu.rows(); jRow++)
        {
            //double dist = EigenHelperFuns::histDistKernel(mu.row(iRow),mu.row(jRow),intervals);
            double dist = EigenHelperFuns::manhattanDist(mu.row(iRow),mu.row(jRow));

            if(dist < 1E-10)
            {
                duplicates.push_back(jRow);
                //        MyHelperFuns::printEigenVec(mu.row(iRow),"Vec1");
                //        MyHelperFuns::printEigenVec(mu.row(jRow),"Vec2");
                //        std::cout << "Distance" << dist << std::endl;
            }
        }
    }

    if(duplicates.size()>0)
    {
        for(std::vector<int>::iterator iter = duplicates.begin(); iter!=duplicates.end();++iter)
            mu.row(*iter) = X.row(MyHelperFuns::randIdx(X.rows()));

        // Do recursive call
        removeCentroidDoubles(mu,X);
    }
}
