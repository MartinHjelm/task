#ifndef PCLTYPEDEFS_H
#define PCLTYPEDEFS_H

// PCL headers
#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>

struct PointT
{
        PCL_ADD_POINT4D;
        PCL_ADD_RGB;  //preferred way of adding a XYZRGB + padding
        int ID;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW //Make sure our new allocators are aligned
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointT, 	
									(float, x, x)
									(float, y, y)
									(float, z, z)
									(float, rgb, rgb)
									(int, ID, ID)
);

// typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal PointN;
typedef pcl::PointCloud<PointT> PC;
typedef pcl::PointCloud<PointN> PCN;

typedef Eigen::Vector3f V3f;
typedef Eigen::Vector4f V4f;

#endif // PCLTYPEDEFS_H
