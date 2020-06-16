#ifndef SYMMETRYCOMPLETION_H
#define SYMMETRYCOMPLETION_H

// PCL
#include "PCLTypedefs.h"

// GSL
#include <gsl/gsl_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>

// Mine
#include "mainaxes.h"



class SymmetryCompletion
{

    PC::Ptr cloud_;
    PCN::Ptr cloudNormals_;
    MainAxes FMA_;
    V3f vp_; // Viewing direction for camera

    struct SymmetryParams
    {
        double bestRot;
        double bestAxisRot;
        double bestTrans;
        double score;
        int axis;
    };

    struct GSLParams
    {
        PC::Ptr cloud_;
        PCN::Ptr cloudNormals_;
        V3f planeNormal_;
        V3f axisrot_;
        V3f midPoint;
    };

    void doSymmetryOptimization(SymmetryParams &sParams, const int &maxIter=100);
    double computeScore( const V3f &midPoint, const V3f &n );

    void computeSymmetryScore(SymmetryParams &sParams);
    void constructCompletion(const SymmetryParams &sParams);


    // GSL stuff
    V3f planeNormal_;
    V3f axisrot_;
    void optUsingGSL(SymmetryParams &sParams);
    static double computeScoreGSL( const gsl_vector *v, void *params );


public:
    SymmetryCompletion();
    void setInputSources(PC::Ptr &cloud, PCN::Ptr &cloudNormals, V3f &vp, const MainAxes &FMA );
    void completeCloud();
};

#endif // SYMMETRYCOMPLETION_H
