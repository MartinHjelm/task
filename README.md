## Disclaimer
Code is messy and needs refactoring, but runs!

## Dependencies 
* PCL
* Boost
* OpenCV
* Eigen
* VTK
* A bunch of other stuff that probaly will give you compilation errors if you haven't installed them.

## How to get the code running
* Install helper_libs(using the sh file) or link them to the CMake file. The helper libs contains some IO and other underlying stuff that the other code relies on.
* Edit CMake files to set directories correctly. 
* There are also sub directories with CMake files that might need to be tuned. 
* Copy contents of the dir copy_content_to_build_dir to the build directory. Contains textfiles with all the parameters.

## Extracting features 
collect_data_full.cpp / collect_data.cpp
The process I used was to have one pcd file with only the object and another with me grasping the object (Paths to those files are given in the beginning of the file.). I then segmented out a bounding box around my grasp and computed an approach vector from the grasp from my arm. I then took the bounding box applied it to the object-only scene and took out the point cloud contained in it. I then computed the features over the point cloud and 2D image inside the bounding box. 

To visualize this you can run this inside the build directory

 ./viewgrasp ../pcds/bottle1.pcd ../pcds/bottle1g2.pcd

which outputs the bounding box and grasp. 

So if all you have is a bounding box with a directional approach vector you can just use the computeFeaturesFromGrasp function that I use further down in the file. 

Note: The SceneSegmentation class segments out an object on a table. But it can also just take a point cloud look in the file viewfeatures.cpp how you can use the setupCloud function of the class if all you have is a point cloud. 

## Computing metric learning transformation
I used the Shogun library which has a LMNN implementation that can use the diagonal. However, any library that use LMNN should be fine. With just using the diagonal you don't learn any correlations between features which can be good if you have very little training data. 

## Not in the paper but in my thesis
viewgraspfactor.cpp
I programmed a grasp synthsizer that finds planes in the point cloud and places a pinch grasp around the plane(Like you are trying to grasp a 2D picture with using thumb and index finger). It then generates a bounding box around that point, computes it features, projects the features from the LMNN transform. It then uses kNN to select if it is a valid graps. I do this for a number of candidates and then select the most probable grasp using KDE(copmuted from the training data). The files are messy and probably does not work since I can't find the original feature files(probably on a drive somewhere) but they could be worth looking around in.


