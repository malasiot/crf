#ifndef __CERTH_DATASET_HPP__
#define __CERTH_DATASET_HPP__

#include "dataset.hpp"


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
