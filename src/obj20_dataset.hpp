#ifndef __OBJ20_DATASET_HPP__
#define __OBJ20_DATASET_HPP__

#include "dataset.hpp"

struct OBJ20_Dataset: public Dataset {
    OBJ20_Dataset(): Dataset() {}

    virtual void loadImages(const std::string &image_path, const PinholeCamera &cam, uint num_images,
                             uint fg_samples_per_image, uint bg_samples_per_image) ;

    virtual void loadBackgroundImages(const std::string &image_path, uint num_images) ;

    virtual void loadModels(const std::string &model_path) ;

private:
    void load_pose(const std::string &info_path, std::string &label) ;
    bool load_camera(const std::string &dir, Eigen::Matrix4f &cam) ;
    bool load_cloud(const std::string &fp, const Eigen::Matrix4f &world_to_model, cvx::util::EPointList3f &cloud);
} ;

#endif
