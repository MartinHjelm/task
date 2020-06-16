#include "featureposesurfaceangle.h"

// STD
#include <vector>
#include <cmath>

// Mine
#include "myhelperfuns.h"
#include "eigenhelperfuns.h"

FeaturePoseSurfaceAngle::FeaturePoseSurfaceAngle()
{}

void
FeaturePoseSurfaceAngle::setInputSource(const PC::Ptr &cldPtr, const PCN::Ptr &cldNrmlPtr)
{
    cloud = cldPtr;
    cloudNormals = cldNrmlPtr;
}

std::vector<double>
FeaturePoseSurfaceAngle::compute(const cuboid &c, const pcl::PointIndices::Ptr &points) const
{
    int Nbins = 21;
    double PI2 = 2*M_PI;
    std::vector<double> ft(Nbins,0.0);
    V3f approachVector = c.axisMat.col(0);
    std::vector<int>::const_iterator iter = points->indices.begin();
    for(; iter!=points->indices.end(); iter++)
    {
//        V3f ptCubeVec = cloud->at(*iter).getVector3fMap() - c.transVec;
//        if(ptCubeVec.norm()==0)
//            continue;

        V3f normalVec = cloudNormals->at(*iter).getNormalVector3fMap();
        // Compute angle between vectors
        double vecAng = std::acos(approachVector.normalized().dot(normalVec.normalized()))/PI2;
        //std::cout << vecAng << ", " << (int)(vecAng/0.1) << std::endl;
//        EigenHelperFuns::printEigenVec(ptCubeVec,"Cube Vec");
//        EigenHelperFuns::printEigenVec(normalVec,"Normal Vec");
//        std::cout << vecAng << " " << (int)(vecAng * Nbins) << std::endl;
        if(vecAng<0)
            vecAng +=0.5;
        int binId = (int)(vecAng * Nbins);
        ft[binId]++;
    }

    // Normalize
    MyHelperFuns::normalizeVec(ft);
    return ft;
}


std::vector<double>
FeaturePoseSurfaceAngle::compute(const V3f &approachVec, const pcl::PointIndices::Ptr &points) const
{
    assert(points->indices.size()!=0);
    assert(approachVec.norm()!=0);

    int Nbins = 21;
    double PI2 = 2*M_PI;
    std::vector<double> ft(Nbins,0.0);

    std::vector<int>::const_iterator iter = points->indices.begin();
    for(; iter!=points->indices.end(); iter++)
    {
//        V3f ptCubeVec = cloud->at(*iter).getVector3fMap() - c.transVec;
//        if(ptCubeVec.norm()==0)
//            continue;

        V3f normalVec = cloudNormals->at(*iter).getNormalVector3fMap();
//        EigenHelperFuns::printEigenVec(normalVec,"NormalVec");
//        EigenHelperFuns::printEigenVec(approachVec,"approachVec");
        // Compute angle between vectors
        double vecAng = std::acos(approachVec.normalized().dot(normalVec.normalized()))/PI2;
        //std::cout << vecAng << ", " << (int)(vecAng/0.1) << std::endl;
//        EigenHelperFuns::printEigenVec(ptCubeVec,"Cube Vec");
//        EigenHelperFuns::printEigenVec(normalVec,"Normal Vec");
//        std::cout << vecAng << " " << (int)(vecAng * Nbins) << std::endl;
        if(vecAng<0)
            vecAng +=0.5;
        int binId = (int)(vecAng * Nbins);
        ft[binId]++;
    }

    // Normalize
    MyHelperFuns::normalizeVec(ft);

//    double sum=0.0;
//    for(int i = 0; i!=ft.size(); i++)
//        sum += ft[i];

//    printf("Final sum %f",sum);

    return ft;
}
