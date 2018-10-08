#ifndef __CERTH_DATASET_HPP__
#define __CERTH_DATASET_HPP__

#include "dataset.hpp"

/*
 *  For each object create a folder with the name of the object and an optional variant e.g. cup_v1/
 *  On this folder put the following files.
 *
 *  box.txt (3D model extents: X Y Z)
 *  w2m.txt (4x4 matrix from world to model coordinates. model coordinates should be -e/2 < x < e/2)
 *  model.obj ( 3D triangle model of the object )
 *  merged.ply (a point cloud of the object e.g. obtained by merging several views or by model sampling)
 *
 *  pose_XXXXX.png: view pose (4x4)
 *  rgb_XXXXX.png:  view rgb image
 *  depth_XXXXX.png: view depth image
 *  mask_XXXXX.png:  view mask (object vs background)
 *
 * where XXXXX is the view id ([0-9]+)
 *
 * Background images are those of the BG_Rooms dataset
 * https://hci.iwr.uni-heidelberg.de/vislearn/wp-content/uploads/2018/03/readme_bg.pdf
 *
 * */

struct CERTH_Dataset: public Dataset {
    CERTH_Dataset(): Dataset() {}

    virtual void loadImages(const std::string &image_path, const PinholeCamera &cam, uint num_images,
                             uint fg_samples_per_image, uint bg_samples_per_image) ;

    virtual void loadBackgroundImages(const std::string &image_path, uint num_images) ;

    virtual void loadModels(const std::string &model_path) ;

private:
    void load_pose(const std::string &path, Eigen::Matrix4f &pose) ;
    bool load_camera(const std::string &dir, Eigen::Matrix4f &cam) ;
    bool load_cloud(const std::string &fp, cvx::util::EPointList3f &cloud);
} ;

#endif
