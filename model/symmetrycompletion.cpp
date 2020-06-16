#include "symmetrycompletion.h"

// STD
#include <vector>

// GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>


// PCL
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

// Mine
#include "eigenhelperfuns.h"
#include "pclhelperfuns.h"
#include <myhelperfuns.h>


SymmetryCompletion::SymmetryCompletion():
    cloud_(new PC),
    cloudNormals_(new PCN)
{}


void
SymmetryCompletion::setInputSources(PC::Ptr &cloud, PCN::Ptr &cloudNormals, V3f &vp, const MainAxes &FMA )
{
    cloud_ = cloud;
    cloudNormals_ = cloudNormals;
//    PCLHelperFuns::computeCloudNormals(cloud,0.03,cloudNormals_);
    FMA_ = FMA;
    vp_ = vp;
}


void
SymmetryCompletion::optUsingGSL(SymmetryParams &sParams)
{

    // Extract inliers from the segmented cloud
    PC::Ptr cloud(new PC);
    PCN::Ptr cloudNormals(new PCN);

    if((double)FMA_.inliers->indices.size()/(double)cloud_->size() > 0.7)
    {
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative (false);
        extract.setInputCloud (cloud_);
        extract.setIndices (FMA_.inliers);
        extract.filter (*cloud);

        pcl::ExtractIndices<PointN> extract_normals;
        extract_normals.setInputCloud (cloudNormals_);
        extract_normals.setIndices (FMA_.inliers);
        extract_normals.filter (*cloudNormals);

        printf("Using only inliers! Ratio: %f",(double)FMA_.inliers->indices.size()/(double)cloud_->size());
    }
    else
    {
        pcl::copyPointCloud(*cloud_,*cloud);
        *cloudNormals = *cloudNormals_;
    }



    // Set axes
    if( std::fabs(vp_.dot(FMA_.axesVectors.col(sParams.axis))) >  std::fabs(vp_.dot(-FMA_.axesVectors.col(sParams.axis))) )
    {
        planeNormal_ = FMA_.axesVectors.col(sParams.axis);
        axisrot_ = FMA_.axesVectors.col(0);
    }
    else
    {
        planeNormal_ = -FMA_.axesVectors.col(sParams.axis);
        axisrot_ = FMA_.axesVectors.col(0);
    }

    planeNormal_.normalize();
    axisrot_.normalize();


//    const gsl_multimin_fminimizer_type *T;
    gsl_multimin_fminimizer *s;
    gsl_multimin_function my_func;






    /** GSL **/

    GSLParams *gslP, gp;
    gp.cloud_ = cloud;
    gp.cloudNormals_ = cloudNormals;
    gp.midPoint = FMA_.midPoint;
    gp.planeNormal_ = planeNormal_;
    gp.axisrot_ = axisrot_;
    gslP = &gp;

    my_func.f = SymmetryCompletion::computeScoreGSL;
    my_func.n = 2;
    my_func.params = gslP;

    gsl_vector *x = gsl_vector_alloc(2);
    gsl_vector_set (x, 0, 0.0); // Translation
    gsl_vector_set (x, 1, 0.0); // Rotation


    gsl_vector *step_size = gsl_vector_alloc(2);
    gsl_vector_set(step_size, 0, 0.1);
    gsl_vector_set(step_size, 1, 5.5);

    s = gsl_multimin_fminimizer_alloc (gsl_multimin_fminimizer_nmsimplex2, 2);
    gsl_multimin_fminimizer_set (s, &my_func, x, step_size);

    size_t iter = 0;
    int status;
    double size;
    do
      {
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);
        if (status)
              break;

        size = gsl_multimin_fminimizer_size (s);
        status = gsl_multimin_test_size (size, 1e-2);

        if (status == GSL_SUCCESS)
           {
             printf ("converged to minimum at\n");
           }

//        printf ("%5lu %10.3e %10.3e f() = %7.3f size = %.3f\n",
//                      iter,
//                      gsl_vector_get (s->x, 0),
//                      gsl_vector_get (s->x, 1),
//                      s->fval, size); fflush(stdout);
      }
    while (status == GSL_CONTINUE && iter < 100);


    sParams.bestTrans = gsl_vector_get (s->x, 0);
    sParams.bestRot = gsl_vector_get (s->x, 1);
//    sParams.bestAxisRot = bestAxisRot;
    sParams.score = s->fval;
    gsl_multimin_fminimizer_free (s);

    printf("Best transform: %f\n",sParams.bestTrans);
    printf("Best rotation: %f\n",sParams.bestRot);
    printf("Best score: %f\n",sParams.score);

}

double
SymmetryCompletion::computeScoreGSL( const gsl_vector *v, void *params )
{


    double translation = gsl_vector_get(v, 0);
    double rotation = gsl_vector_get(v, 1);

    GSLParams *gslP = (GSLParams *)params;

    PC::Ptr cloud = (*gslP).cloud_;
    PCN::Ptr cloudNormals = (*gslP).cloudNormals_;
    V3f midPointConst = (*gslP).midPoint;
    V3f n = (*gslP).planeNormal_;
    V3f axisrot = (*gslP).axisrot_;

    // Translation of midpoint
    V3f midPoint = midPointConst + translation*n;

    // Plane rotation
//    EigenHelperFuns::rotVecDegAroundAxis(n,axisRotation,axisrot);
    EigenHelperFuns::rotVecDegAroundAxis(axisrot,rotation,n);



    // 2. Compute reflection
    double b = midPoint.dot(n);
    double E1 = 0;
    int E1counter = 0;
    double E2 = 0;
    int E2counter = 0;
    double E3 = 0;
    int E3counter = 0;

    PC::Ptr rCloud1(new PC);
    PC::Ptr rCloud2(new PC);
    PCN::Ptr rCloudNormals(new PCN);
    for( uint idx = 0; idx != cloud->size(); idx++ )
    {
        V3f p = cloud->at(idx).getVector3fMap();
        V3f q = p - 2 * n * ( p.dot(n) - b);
        PointT pt;
        pt = cloud->at(idx);
        pt.getVector3fMap() = q;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        pt.a = 255;
        rCloud1->push_back(pt);

        //                // 3. Project onto image coordinates
        int row, col;
        PCLHelperFuns::projectPt2Img(pt,row,col);

        //                // 4. Compute scores
        if(PCLHelperFuns::isPxInCloud(cloud,row*640+col,p))
        {
            //                    if( std::fabs(q.dot(vp)) < std::fabs(p.dot(vp)) )
            if( q.norm() < p.norm() )
            {
                //                        E1 =  std::fabs(p.dot(vp)) - std::fabs(q.dot(vp));
                E1 += (q-p).norm();
                E1counter++;
            }
            else
            {rCloud2->push_back(pt);}

        }
        else
        {
            double dBest = 1E10;
            for( uint idx = 0; idx != cloud->size(); idx++ )
            {
                int y = cloud->at(idx).ID/640;
                int x = cloud->at(idx).ID%640;

                // Euclidian
                //                        double d = std::sqrt( std::pow((double)(y-row),2) + std::pow((double)(x-col),2) );
                // City block
                double d = std::fabs((double)(y-row)) + std::fabs((double)(x-col));

                if(d<dBest)
                    dBest = d;
            }
            E2 += dBest;
            E2counter++;
        }
    }


    // Compare cloud normals
    PCLHelperFuns::computeCloudNormals(rCloud1,0.01,rCloudNormals);
    for( uint id = 0; id != rCloudNormals->size(); id++ )
    {
        V3f pNormal = cloudNormals->at(id).getNormalVector3fMap();
        V3f qNormal = rCloudNormals->at(id).getNormalVector3fMap();
        double angle = std::acos(pNormal.dot(qNormal)/(pNormal.norm()*qNormal.norm()));
        angle = (std::min) (angle, M_PI - angle);
        angle = RAD2DEG(angle);
        if(angle > 1.0 )
        {
            E3 += angle;
            E3counter++;
        }

    }


    // Compare average distance to plane
    double dC = 0.0;
    double dR = 0.0;
    double dCcount = 0.0;
    double dRcount = 0.0;
    std::vector<double> dVec1, dVec2, hist1, hist2;

    for( uint id = 0; id != cloud->size(); id++ )
    {
        V3f p = cloud->at(id).getVector3fMap();
        double dp = n.dot(p-midPoint);
        if(dp>0)
        {
            dR += std::fabs(dp);
            dRcount++;
            dVec1.push_back(std::fabs(dp));
        }
        else
        {
            dC += std::fabs(dp);
            dCcount++;
            dVec2.push_back(std::fabs(dp));
        }
    }

    for( uint id = 0; id != rCloud2->size(); id++ )
    {
        V3f p = rCloud2->at(id).getVector3fMap();
        double dp = n.dot(p-midPoint);
        if(dp>0)
        {
            dR += std::fabs(dp);
            dRcount++;
            dVec1.push_back(std::fabs(dp));
        }
        else
        {
            dC += std::fabs(dp);
            dCcount++;
            dVec2.push_back(std::fabs(dp));
        }
    }

    if(dRcount>0)
        dR /= dRcount;
    if(dCcount>0)
        dC /= dCcount;
    std::cout << dVec2.size() << std::endl;
    MyHelperFuns::vec2Hist(dVec1,11,hist1);
    MyHelperFuns::vec2Hist(dVec2,11,hist2);

    //            MyHelperFuns::printVector(hist1,"Histogram 1 ");
    //            MyHelperFuns::printVector(hist2,"Histogram 2 ");

    double dHist = MyHelperFuns::manhattanDist(hist1,hist2);

    //            double E4 =0.0;// std::fabs(dR-dC);
    //            double E4 = std::fabs(dRcount-dCcount);
    double E4 = 1E1*dHist;
    double E4counter = 1;


    //            // 4. Compute score
    double score = 0.0;
    if(E1counter > 0 || E2counter > 0 || E3counter)
    {
        if(E1counter>0)
        {
            score += E1/E1counter;
//            printf(" E1: %f",E1/E1counter);
        }
        if(E2counter>0)
        {
            score += 0.5f * (E2/E2counter);
//            printf(" E2: %f",0.5f * (E2/E2counter));
        }
        if(E3counter>0)
        {
            score += 1.f * (E3/E3counter);
//            printf(" E3: %f",1.f * (E3/E3counter));
        }
        if(E4counter>0)
        {
            score += E4;
//            printf(" E4: %f",E4);
        }
    }
    else
    {
        score = 1E10;
    }

//    printf(" Score: %f \n",score);

    return score;

}



void
SymmetryCompletion::completeCloud()
{
    printf("Starting symmetry computations\n");fflush(stdout);

    SymmetryParams p1;
    SymmetryParams p2;

    p1.axis = 1;
    p2.axis = 2;

    // Compute score for the two axes
//    doSymmetryOptimization(p1);
//    constructCompletion(p1);

//    doSymmetryOptimization(p2);
//    constructCompletion(p2);

    optUsingGSL(p1);
    optUsingGSL(p2);

    if(p1.score>p2.score)
    {
        constructCompletion(p1);
//        optUsingGSL(p2);
        constructCompletion(p2);
    }
    else
    {
        constructCompletion(p2);
//        optUsingGSL(p1);
        constructCompletion(p1);

    }

//    constructCompletion(p2);


printf("Symmetry computations done.\n");fflush(stdout);
}


void
SymmetryCompletion::computeSymmetryScore(SymmetryParams &sParams)
{
    V3f planeNormal;
    V3f axisrot;

    if( std::fabs(vp_.dot(FMA_.axesVectors.col(sParams.axis))) <  std::fabs(vp_.dot(-FMA_.axesVectors.col(sParams.axis))) )
    {
        planeNormal = FMA_.axesVectors.col(sParams.axis);
        axisrot = FMA_.axesVectors.col(0);
    }
    else
    {
        planeNormal = -FMA_.axesVectors.col(sParams.axis);
        axisrot = FMA_.axesVectors.col(0);
    }

    planeNormal.normalize();
    axisrot.normalize();

    double bestTrans = 0.0;
    double bestRot = 0.0;
    double bestScore = 1E10;

    double rotRange = 10;//10
    double transRange = 10  ; //10
    double degree = 5.0;
    double displacement = 0.005;

    // For every reflection plane roation and translation compute score
    for(double i_trans=-transRange; i_trans<=transRange; i_trans++)
    {
        for(double i_rot=-rotRange; i_rot<=rotRange; i_rot++)
        {
//            printf("Translation: %f Rotation: %f",i_trans*displacement,i_rot*degree);

            // Translation of midpoint
            V3f midPoint = FMA_.midPoint + i_trans*displacement*planeNormal;
            V3f n = planeNormal;

            // Plane rotation
            EigenHelperFuns::rotVecDegAroundAxis(axisrot,i_rot*degree, n);

            // 2. Compute reflection
            double b = midPoint.dot(n);
            double E1 = 0;
            int E1counter = 0;
            double E2 = 0;
            int E2counter = 0;
            double E3 = 0;
            int E3counter = 0;

            PC::Ptr rCloud(new PC);
            PCN::Ptr rCloudNormals(new PCN);
            for( uint idx = 0; idx != cloud_->size(); idx++ )
            {
                V3f p = cloud_->at(idx).getVector3fMap();
                V3f q = p - 2 * n * ( p.dot(n) - b);
                PointT pt;
                pt = cloud_->at(idx);
                pt.getVector3fMap() = q;
                pt.r = 255;
                pt.g = 0;
                pt.b = 0;
                pt.a = 255;
                rCloud->push_back(pt);

                //                // 3. Project onto image coordinates
                int row, col;
                PCLHelperFuns::projectPt2Img(pt,row,col);

                //                // 4. Compute scores
                if(PCLHelperFuns::isPxInCloud(cloud_,row*640+col,p))
                {
                    //                    if( std::fabs(q.dot(vp)) < std::fabs(p.dot(vp)) )
                    if( q.norm() < p.norm() )
                    {
                        //                        E1 =  std::fabs(p.dot(vp)) - std::fabs(q.dot(vp));
                        E1 += (q-p).norm();
                        E1counter++;
                    }
                }
                else
                {
                    double dBest = 1E10;
                    for( uint idx = 0; idx != cloud_->size(); idx++ )
                    {
                        int y = cloud_->at(idx).ID/640;
                        int x = cloud_->at(idx).ID%640;

                        // Euclidian
                        //                        double d = std::sqrt( std::pow((double)(y-row),2) + std::pow((double)(x-col),2) );
                        // City block
                        double d = std::fabs((double)(y-row)) + std::fabs((double)(x-col));

                        if(d<dBest)
                            dBest = d;
                    }
                    E2 += dBest;
                    E2counter++;
                }
            }



            PCLHelperFuns::computeCloudNormals(rCloud,0.01,rCloudNormals);
            for( uint id = 0; id != rCloudNormals->size(); id++ )
            {
                V3f pNormal = cloudNormals_->at(id).getNormalVector3fMap();
                V3f qNormal = rCloudNormals->at(id).getNormalVector3fMap();
                double angle = std::acos(pNormal.dot(qNormal)/(pNormal.norm()*qNormal.norm()));
                angle = (std::min) (angle, M_PI - angle);
                angle = RAD2DEG(angle);
                if(angle > 1.0 )
                {
                    E3 += angle;
                    E3counter++;
                }

            }
            //            // 4. Compute score
            double score = 0.0;
            if(E1counter > 0 || E2counter > 0 || E3counter)
            {
                if(E1counter>0)
                {
                    score += E1/E1counter;
//                    printf(" E1: %f",E1/E1counter);
                }
                if(E2counter>0)
                {
                    score += 0.5f * (E2/E2counter);
//                    printf(" E2: %f",0.5f * (E2/E2counter));
                }
                if(E3counter>0)
                {
                    score += E3/E3counter;
//                    printf(" E3: %f",E3/E3counter);
                }
            }
            else
            {
                score = 1E10;
            }

//            printf(" Score: %f \n",score);

            if(score < bestScore)
            {
                bestScore = score;
                bestRot = i_rot*degree;
                bestTrans = i_trans*displacement;
            }

        }
    }

    printf("Best transform: %f\n",bestTrans);
    printf("Best rotation: %f\n",bestRot);
    printf("Best score: %f\n",bestScore);

    sParams.bestTrans = bestTrans;
    sParams.bestRot = bestRot;
    sParams.score = bestScore;
}



void
SymmetryCompletion::doSymmetryOptimization(SymmetryParams &sParams, const int &maxIter)
{
    V3f planeNormal;
    V3f axisrot;

    if( std::fabs(vp_.dot(FMA_.axesVectors.col(sParams.axis))) <  std::fabs(vp_.dot(-FMA_.axesVectors.col(sParams.axis))) )
    {
        planeNormal = FMA_.axesVectors.col(sParams.axis);
        axisrot = FMA_.axesVectors.col(0);
    }
    else
    {
        planeNormal = -FMA_.axesVectors.col(sParams.axis);
        axisrot = FMA_.axesVectors.col(0);
    }

    planeNormal.normalize();
    axisrot.normalize();

    double bestTrans = 0.0;
    double bestRot = 0.0;
    double bestAxisRot = 0.0;
    double bestScore = 1E10;

    double rotRange = 1;
    double transRange = 1;
    double degree = 5.0;
    double axisdegree = 5.0;
    double displacement = 0.01;
    int counter = 0;
    // For every reflection plane roation and translation compute score
    while(counter<maxIter)
    {

        std::vector<double> scores;
        std::vector<std::vector<double> > mods;

        for(double i_rotAxis=-rotRange; i_rotAxis<=rotRange; i_rotAxis++)
        {
        for(double i_trans=-transRange; i_trans<=transRange; i_trans++)
        {
            for(double i_rot=-rotRange; i_rot<=rotRange; i_rot++)
            {

                double translation = i_trans*displacement+bestTrans;
                double rotation = bestRot+i_rot*degree;
                double axisRotation = bestAxisRot+i_rotAxis*axisdegree;

                printf("Translation: %f Rotation: %f AxisRotation: %f ",translation,rotation,axisRotation);

                // Translation of midpoint
                V3f midPoint = FMA_.midPoint + translation*planeNormal;
                V3f n = planeNormal;

                // Plane rotation
//                EigenHelperFuns::rotVecDegAroundAxis(n,axisRotation,axisrot);
                EigenHelperFuns::rotVecDegAroundAxis(axisrot,rotation,n);


                // Compute score
                scores.push_back(computeScore(midPoint,n));
                std::vector<double> transform;
                transform.push_back(translation);
                transform.push_back(rotation);
                transform.push_back(axisRotation);
                mods.push_back(transform);
            }
        }
        }

        int minIdx = MyHelperFuns::minIdxOfVector(scores);
        if(scores[minIdx] < bestScore )
        {
            bestTrans = mods[minIdx][0];;
            bestRot = mods[minIdx][1];
            bestAxisRot = mods[minIdx][2];
            bestScore = scores[minIdx];
        }
        else if(std::fabs(degree-.5)>1E-6)
        {
            printf("Decreasing degree direction.\n");
            degree-=0.5;
        }
        else if(std::fabs(axisdegree-.5)>1E-6)
        {
            printf("Decreasing degree direction.\n");
            axisdegree-=0.5;
        }
        else if(std::fabs(displacement-0.005)>1E-6)
        {
            printf("Decreasing translation direction.\n");
            displacement -= 0.005;
        }
        else
            break;

        printf("Best score: %f\n",bestScore);

        counter++;
    }

    sParams.bestTrans = bestTrans;
    sParams.bestRot = bestRot;
    sParams.bestAxisRot = bestAxisRot;
    sParams.score = bestScore;


}


double
SymmetryCompletion::computeScore( const V3f &midPoint, const V3f &n )
{

    // 2. Compute reflection
    double b = midPoint.dot(n);
    double E1 = 0;
    int E1counter = 0;
    double E2 = 0;
    int E2counter = 0;
    double E3 = 0;
    int E3counter = 0;

    PC::Ptr rCloud(new PC);
    PCN::Ptr rCloudNormals(new PCN);
    for( uint idx = 0; idx != cloud_->size(); idx++ )
    {
        V3f p = cloud_->at(idx).getVector3fMap();
        V3f q = p - 2 * n * ( p.dot(n) - b);
        PointT pt;
        pt = cloud_->at(idx);
        pt.getVector3fMap() = q;
        pt.r = 255;
        pt.g = 0;
        pt.b = 0;
        pt.a = 255;
        rCloud->push_back(pt);

        //                // 3. Project onto image coordinates
        int row, col;
        PCLHelperFuns::projectPt2Img(pt,row,col);

        //                // 4. Compute scores
        if(PCLHelperFuns::isPxInCloud(cloud_,row*640+col,p))
        {
            //                    if( std::fabs(q.dot(vp)) < std::fabs(p.dot(vp)) )
            if( q.norm() < p.norm() )
            {
                //                        E1 =  std::fabs(p.dot(vp)) - std::fabs(q.dot(vp));
                E1 += (q-p).norm();
                E1counter++;
            }

        }
        else
        {
            double dBest = 1E10;
            for( uint idx = 0; idx != cloud_->size(); idx++ )
            {
                int y = cloud_->at(idx).ID/640;
                int x = cloud_->at(idx).ID%640;

                // Euclidian
                //                        double d = std::sqrt( std::pow((double)(y-row),2) + std::pow((double)(x-col),2) );
                // City block
                double d = std::fabs((double)(y-row)) + std::fabs((double)(x-col));

                if(d<dBest)
                    dBest = d;
            }
            E2 += dBest;
            E2counter++;
        }
    }


    // Compare cloud normals
    PCLHelperFuns::computeCloudNormals(rCloud,0.01,rCloudNormals);
    for( uint id = 0; id != rCloudNormals->size(); id++ )
    {
        V3f pNormal = cloudNormals_->at(id).getNormalVector3fMap();
        V3f qNormal = rCloudNormals->at(id).getNormalVector3fMap();
        double angle = std::acos(pNormal.dot(qNormal)/(pNormal.norm()*qNormal.norm()));
        angle = (std::min) (angle, M_PI - angle);
        angle = RAD2DEG(angle);
        if(angle > 1.0 )
        {
            E3 += angle;
            E3counter++;
        }

    }


    // Compare average distance to plane
    double dC = 0.0;
    double dR = 0.0;
    double dCcount = 0.0;
    double dRcount = 0.0;
    std::vector<double> dVec1, dVec2, hist1, hist2;

    for( uint id = 0; id != cloud_->size(); id++ )
    {
        V3f p = cloud_->at(id).getVector3fMap();
        double dp = n.dot(p-midPoint);
        if(dp>0)
        {
            dR += std::fabs(dp);
            dRcount++;
            dVec1.push_back(std::fabs(dp));
        }
        else
        {
            dC += std::fabs(dp);
            dCcount++;
            dVec2.push_back(std::fabs(dp));
        }
    }

    //            for( int id = 0; id != rCloud->size(); id++ )
    //            {
    //                V3f p = rCloud->at(id).getVector3fMap();
    //                double dp = n.dot(p-midPoint);
    //                if(dp>0)
    //                {
    //                    dR += std::fabs(dp);
    //                    dRcount++;
    //                    dVec1.push_back(std::fabs(dp));
    //                }
    //                else
    //                {
    //                    dC += std::fabs(dp);
    //                    dCcount++;
    //                    dVec2.push_back(std::fabs(dp));
    //                }
    //            }

    if(dRcount>0)
        dR /= dRcount;
    if(dCcount>0)
        dC /= dCcount;

    MyHelperFuns::vec2Hist(dVec1,99,hist1);
    MyHelperFuns::vec2Hist(dVec2,99,hist2);

    //            MyHelperFuns::printVector(hist1,"Histogram 1 ");
    //            MyHelperFuns::printVector(hist2,"Histogram 2 ");

    double dHist = MyHelperFuns::manhattanDist(hist1,hist2);

    //            double E4 =0.0;// std::fabs(dR-dC);
    //            double E4 = std::fabs(dRcount-dCcount);
    double E4 = 1E0*dHist;
    double E4counter = 1;




    //            // 4. Compute score
    double score = 0.0;
    if(E1counter > 0 || E2counter > 0 || E3counter)
    {
        if(E1counter>0)
        {
            score += E1/E1counter;
                                printf(" E1: %f",E1/E1counter);
        }
        if(E2counter>0)
        {
            score += 0.5f * (E2/E2counter);
                                printf(" E2: %f",0.5f * (E2/E2counter));
        }
        if(E3counter>0)
        {
            score += 1.f * (E3/E3counter);
                                printf(" E3: %f",1.f * (E3/E3counter));
        }
        if(E4counter>0)
        {
            score += E4;
                                printf(" E4: %f",E4);
        }
    }
    else
    {
        score = 1E10;
    }

    printf(" Score: %f \n",score);

    return score;

}




void
SymmetryCompletion::constructCompletion(const SymmetryParams &sParams)
{
    V3f planeNormal, axisrot;

    if( std::fabs(vp_.dot(FMA_.axesVectors.col(sParams.axis))) <  std::fabs(vp_.dot(-FMA_.axesVectors.col(sParams.axis))) )
    {
        planeNormal = FMA_.axesVectors.col(sParams.axis);
        axisrot = FMA_.axesVectors.col(0);
    }
    else
    {
        planeNormal = -FMA_.axesVectors.col(sParams.axis);
        axisrot = FMA_.axesVectors.col(0);
    }

    planeNormal.normalize();
    axisrot.normalize();


    // Compute best rotation and translation
    // Translation of midpoint
    V3f midPoint = FMA_.midPoint + sParams.bestTrans*planeNormal;
    // Plane rotation
    EigenHelperFuns::rotVecDegAroundAxis(axisrot,sParams.bestRot,planeNormal);
    double b = midPoint.dot(planeNormal);

    // 2. Compute reflection
    PC::Ptr cloud(new PC);
    if((double)FMA_.inliers->indices.size()/(double)cloud_->size() > 0.7)
    {
        pcl::ExtractIndices<PointT> extract;
        extract.setNegative (false);
        extract.setInputCloud (cloud_);
        extract.setIndices (FMA_.inliers);
        extract.filter (*cloud);
    }
    else
    {
        cloud = cloud_;
    }


    PC::Ptr cloudQ(new PC);
    for( uint idx = 0; idx != cloud->size(); idx++ )
    {
        V3f p = cloud->at(idx).getVector3fMap();
        V3f q = p - 2 * planeNormal * ( p.dot(planeNormal) - b);
        PointT pt;
        pt = cloud->at(idx);
        pt.getVector3fMap() = q;
        int row, col;
        PCLHelperFuns::projectPt2Img(pt,row,col);
        if(PCLHelperFuns::isPxInCloud(cloud_,row*640+col,p))
        {
            if( q.norm() - p.norm() > 0.01 )
            {
//                EigenHelperFuns::printEigenVec(q);
                pt.ID = row*640+col;
//                pt.r = 255; pt.g = 0; pt.b = 0;
                cloudQ->push_back(pt);
            }
        }
    }

    *cloud_ += *cloudQ;

    // After adding points to cloud recompute the normalspho
    cloudNormals_->clear();
    PCLHelperFuns::smoothCloud(cloud_);
    PCLHelperFuns::computeCloudNormals(cloud_,0.03,cloudNormals_);
}
