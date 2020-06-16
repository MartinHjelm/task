#include "sac3dcirclewithnormal.h"
#include <algorithm>    // std::min


SAC3DCircleWithNormal::SAC3DCircleWithNormal(PC::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr cloudNormals, pcl::PointIndices::Ptr inliers) :
    cloud(cloud),
    cloudNormals(cloudNormals),
    indices_(inliers)
{
    radius_min_ = 0.02;
    radius_max_ = 0.1;
}


bool
SAC3DCircleWithNormal::isSampleGood (
        const std::vector<int> &samples, const V3f &axis)
{
    // Get the values at the three points
    //  Eigen::Vector3f v0 = cloud->points[samples[0]].getArray3fMap();
    //  Eigen::Vector3f v1 = cloud->points[samples[1]].getArray3fMap();
    //  Eigen::Vector3f v2 = cloud->points[samples[2]].getArray3fMap();

    //  // calculate vectors between points
    //  v1 -= v0;
    //  v2 -= v0;
    //  v1 = -v1;
    //  v2 = -v2;

    V3f v1 = cloud->points[samples[0]].getArray3fMap() - cloud->points[samples[1]].getArray3fMap();
    V3f v2 = cloud->points[samples[1]].getArray3fMap() - cloud->points[samples[2]].getArray3fMap();

    // Compute normal
    Eigen::Vector3f n = v1.cross(v2);
    n.normalize();

    V3f axisNormalized = axis.normalized();

    // Compute angle between normal and the wanted normal
    double angle_diff = std::acos(n.dot(axisNormalized));
    angle_diff = (std::min) (angle_diff, M_PI - angle_diff);
    // Check whether the current plane model satisfies our angle threshold criterion with respect to the given axis
    if (angle_diff < DEG2RAD(3))
        return (true);
    else
        return (false);
}



bool
SAC3DCircleWithNormal::isSampleGood2 (Eigen::VectorXf &model_coefficients, const V3f &axis, const V3f &center, const double &maxRadius )
{

    if ( model_coefficients[3] > maxRadius)
    {
//        printf("Radius fail: %f, %f",model_coefficients[3],maxRadius);
        return false;
    }

    V3f n;
    n(0) = model_coefficients[4];
    n(1) = model_coefficients[5];
    n(2) = model_coefficients[6];
    V3f axisNormalized = axis.normalized();
    // Compute angle between normal and the wanted normal
    double dp = n.dot(axisNormalized);
    double angle_diff = std::acos(dp);
    angle_diff = (std::min) (angle_diff, M_PI - angle_diff);

    V3f cc;
    cc(0) = model_coefficients[0];
    cc(1) = model_coefficients[1];
    cc(2) = model_coefficients[2];


    V3f voccc = (center - cc); voccc.normalize();
    double prj  = std::acos(voccc.dot(axisNormalized));
    prj = (std::min) (prj, M_PI - prj);


    // Check whether the current plane model satisfies our angle threshold criterion with respect to the given axis
    if( angle_diff < DEG2RAD(3) && prj < DEG2RAD(20) )
        return (true);
    else
        return (false);
}


bool
SAC3DCircleWithNormal::computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients)
{
    // Need 3 samples
    if (samples.size () != 3)
    {
        PCL_ERROR ("[SAC3DCircleWithNormal::computeModelCoefficients] Invalid set of samples given (%zu)!\n", samples.size ());
        return (false);
    }

    model_coefficients.resize (7);   //needing 7 coefficients: centerX, centerY, centerZ, radius, normalX, normalY, normalZ

    Eigen::Vector3f p0 = cloud->points[samples[0]].getArray3fMap();
    Eigen::Vector3f p1 = cloud->points[samples[1]].getArray3fMap();
    Eigen::Vector3f p2 = cloud->points[samples[2]].getArray3fMap();

    Eigen::Vector3f helper_vec01 = p0 - p1;
    Eigen::Vector3f helper_vec02 = p0 - p2;
    Eigen::Vector3f helper_vec10 = p1 - p0;
    Eigen::Vector3f helper_vec12 = p1 - p2;
    Eigen::Vector3f helper_vec20 = p2 - p0;
    Eigen::Vector3f helper_vec21 = p2 - p1;

    Eigen::Vector3f common_helper_vec = helper_vec01.cross (helper_vec12);

    double commonDividend = 2.0 * common_helper_vec.squaredNorm ();

    double alpha = (helper_vec12.squaredNorm () * helper_vec01.dot (helper_vec02)) / commonDividend;
    double beta =  (helper_vec02.squaredNorm () * helper_vec10.dot (helper_vec12)) / commonDividend;
    double gamma = (helper_vec01.squaredNorm () * helper_vec20.dot (helper_vec21)) / commonDividend;

    Eigen::Vector3f circle_center = alpha * p0 + beta * p1 + gamma * p2;

    Eigen::Vector3f circle_radiusVector = circle_center - p0;
    double circle_radius = circle_radiusVector.norm ();
    Eigen::Vector3f circle_normal = common_helper_vec.normalized ();

    model_coefficients[0] =  (circle_center[0]);
    model_coefficients[1] =  (circle_center[1]);
    model_coefficients[2] =  (circle_center[2]);
    model_coefficients[3] =  (circle_radius);
    model_coefficients[4] =  (circle_normal[0]);
    model_coefficients[5] =  (circle_normal[1]);
    model_coefficients[6] =  (circle_normal[2]);

    return (true);
}


void
SAC3DCircleWithNormal::getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances)
{
    // Check if the model is valid given the user constraints
    if (!isModelValid (model_coefficients))
    {
        distances.clear ();
        return;
    }
    distances.resize (indices_->indices.size ());

    // Iterate through the 3d points and calculate the distances from them to the sphere
    for (size_t i = 0; i < indices_->indices.size (); ++i)
        // Calculate the distance from the point to the circle:
        // 1.   calculate intersection point of the plane in which the circle lies and the
        //      line from the sample point with the direction of the plane normal (projected point)
        // 2.   calculate the intersection point of the line from the circle center to the projected point
        //      with the circle
        // 3.   calculate distance from corresponding point on the circle to the sample point
    {
        // what i have:
        // P : Sample Point
        Eigen::Vector3d P (cloud->points[indices_->indices[i]].x, cloud->points[indices_->indices[i]].y, cloud->points[indices_->indices[i]].z);
        // C : Circle Center
        Eigen::Vector3d C (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
        // N : Circle (Plane) Normal
        Eigen::Vector3d N (model_coefficients[4], model_coefficients[5], model_coefficients[6]);
        // r : Radius
        double r = model_coefficients[3];

        Eigen::Vector3d helper_vectorPC = P - C;
        // 1.1. get line parameter
        double lambda = (helper_vectorPC.dot (N)) / N.squaredNorm ();

        // Projected Point on plane
        Eigen::Vector3d P_proj = P + lambda * N;
        Eigen::Vector3d helper_vectorP_projC = P_proj - C;

        // K : Point on Circle
        Eigen::Vector3d K = C + r * helper_vectorP_projC.normalized ();
        Eigen::Vector3d distanceVector =  P - K;

        distances[i] = distanceVector.norm ();
    }
}


void
SAC3DCircleWithNormal::selectWithinDistance (
        const Eigen::VectorXf &model_coefficients, const double threshold,
        std::vector<int> &inliers)
{
    // Check if the model is valid given the user constraints
    if (!isModelValid (model_coefficients))
    {
        inliers.clear ();
        return;
    }
    int nr_p = 0;
    inliers.resize (cloud->points.size ());

    // Iterate through the 3d points and calculate the distances from them to the sphere
    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        // what i have:
        // P : Sample Point
        Eigen::Vector3d P (cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        // C : Circle Center
        Eigen::Vector3d C (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
        // N : Circle (Plane) Normal
        Eigen::Vector3d N (model_coefficients[4], model_coefficients[5], model_coefficients[6]);
        // r : Radius
        double r = model_coefficients[3];

        Eigen::Vector3d helper_vectorPC = P - C;
        // 1.1. get line parameter
        double lambda = (-(helper_vectorPC.dot (N))) / N.dot (N);
        // Projected Point on plane
        Eigen::Vector3d P_proj = P + lambda * N;
        Eigen::Vector3d helper_vectorP_projC = P_proj - C;

        // K : Point on Circle
        Eigen::Vector3d K = C + r * helper_vectorP_projC.normalized ();
        Eigen::Vector3d distanceVector =  P - K;

        if (distanceVector.norm () < threshold)
        {
            // Returns the indices of the points whose distances are smaller than the threshold
            inliers[nr_p] = i;
            nr_p++;
        }
    }
    inliers.resize (nr_p);
}


int
SAC3DCircleWithNormal::countWithinDistance (
        const Eigen::VectorXf &model_coefficients, const double threshold)
{
    // Check if the model is valid given the user constraints
    if (!isModelValid (model_coefficients))
        return (0);
    int nr_p = 0;

    // Iterate through the 3d points and calculate the distances from them to the sphere
    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        // what i have:
        // P : Sample Point
        Eigen::Vector3d P (cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        // C : Circle Center
        Eigen::Vector3d C (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
        // N : Circle (Plane) Normal
        Eigen::Vector3d N (model_coefficients[4], model_coefficients[5], model_coefficients[6]);
        // r : Radius
        double r = model_coefficients[3];

        Eigen::Vector3d helper_vectorPC = P - C;
        // 1.1. get line parameter
        double lambda = (-(helper_vectorPC.dot (N))) / N.dot (N);

        // Projected Point on plane
        Eigen::Vector3d P_proj = P + lambda * N;
        Eigen::Vector3d helper_vectorP_projC = P_proj - C;

        // K : Point on Circle
        Eigen::Vector3d K = C + r * helper_vectorP_projC.normalized ();
        Eigen::Vector3d distanceVector =  P - K;

        if (distanceVector.norm () < threshold)
            nr_p++;
    }
    return (nr_p);
}


bool
SAC3DCircleWithNormal::doSamplesVerifyModel (
        const std::set<int> &indices,
        const Eigen::VectorXf &model_coefficients,
        const double threshold)
{
    // Needs a valid model coefficients
    if (model_coefficients.size () != 7)
    {
        PCL_ERROR ("[SAC3DCircleWithNormal::doSamplesVerifyModel] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
        return (false);
    }

    for (std::set<int>::const_iterator it = indices.begin (); it != indices.end (); ++it)
    {
        // Calculate the distance from the point to the sphere as the difference between
        //dist(point,sphere_origin) and sphere_radius

        // what i have:
        // P : Sample Point
        Eigen::Vector3d P (cloud->points[*it].x, cloud->points[*it].y, cloud->points[*it].z);
        // C : Circle Center
        Eigen::Vector3d C (model_coefficients[0], model_coefficients[1], model_coefficients[2]);
        // N : Circle (Plane) Normal
        Eigen::Vector3d N (model_coefficients[4], model_coefficients[5], model_coefficients[6]);
        // r : Radius
        double r = model_coefficients[3];
        Eigen::Vector3d helper_vectorPC = P - C;
        // 1.1. get line parameter
        double lambda = (-(helper_vectorPC.dot (N))) / N.dot (N);
        // Projected Point on plane
        Eigen::Vector3d P_proj = P + lambda * N;
        Eigen::Vector3d helper_vectorP_projC = P_proj - C;

        // K : Point on Circle
        Eigen::Vector3d K = C + r * helper_vectorP_projC.normalized ();
        Eigen::Vector3d distanceVector =  P - K;

        if (distanceVector.norm () > threshold)
            return (false);
    }
    return (true);
}


bool
SAC3DCircleWithNormal::isModelValid (const Eigen::VectorXf &model_coefficients)
{
    // Needs a valid model coefficients
    if (model_coefficients.size () != 7)
    {
        PCL_ERROR ("[SAC3DCircleWithNormal::isModelValid] Invalid number of model coefficients given (%zu)!\n", model_coefficients.size ());
        return (false);
    }

    if (model_coefficients[3] < radius_min_)
        return (false);
    if (model_coefficients[3] > radius_max_)
        return (false);

    return (true);
}



