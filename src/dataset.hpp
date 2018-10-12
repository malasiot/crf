#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <Eigen/Geometry>

#include <cvx/util/camera/camera.hpp>
#include <cvx/util/misc/binary_stream.hpp>
#include <cvx/util/geometry/point_list.hpp>
#include <cvx/util/math/rng.hpp>
#include <cvx/viz/scene/scene.hpp>

typedef uint64_t sample_idx_t ;
typedef uint32_t node_idx_t ;
typedef std::vector<sample_idx_t> subset_t ;

using cvx::util::IBinaryStream ;
using cvx::util::OBinaryStream ;
using cvx::util::PinholeCamera ;
using cvx::util::RNG ;

struct Dataset {

    Dataset() ;

    uint64_t numSamples() const { return samples_.size() ; }

    const cv::Mat &getDepthImageD(sample_idx_t idx) const {
        return ( labels_[idx].empty() ) ? planar_images_[image_idx_[idx]] : depth_images_[image_idx_[idx]] ;
    }

    const cv::Mat &getDepthImageC(sample_idx_t idx) const {
        return ( labels_[idx].empty() ) ? bg_depth_images_[bg_image_idx_[idx]] : depth_images_[image_idx_[idx]] ;
    }
    const cv::Mat &getColorImage(sample_idx_t idx) const {
        return ( labels_[idx].empty() ) ? bg_rgb_images_[bg_image_idx_[idx]] : rgb_images_[image_idx_[idx]];
    }

    const cv::Point getPoint(sample_idx_t idx) const {
        return samples_[idx] ;
    }

    const Eigen::Vector3f getCoordinates(sample_idx_t idx) const {
        return coordinates_[idx] ;
    }

    virtual void loadImages(const std::string &image_path, const PinholeCamera &cam, uint max_num_images_per_id,
                    uint fg_samples_per_image, uint bg_samples_per_image) {}

    virtual void loadBackgroundImages(const std::string &image_path, uint num_images) {}

    virtual void loadModels(const std::string &model_path) {}

    void addSingleImage(const cv::Mat &c, const cv::Mat &d, const cv::Mat &m, uint samples) ;

    static cv::Mat makePlanarDepth(const PinholeCamera &cam, const Eigen::Matrix4f &pose, const Eigen::Vector3f &box) ;


    RNG rng_ ;
    std::vector<cv::Point> samples_ ;  // the list of point samples (coming from several objects and views)
    std::vector<cv::Mat> depth_images_, rgb_images_, masks_ ; // the list of images used for sampling pixels
    std::vector<cv::Mat> bg_depth_images_, bg_rgb_images_, planar_images_ ; // the list of images used for sampling pixels
    std::vector<Eigen::Matrix4f> poses_ ; // poses associated with each view
    std::vector<uint32_t> image_idx_ ; // this maps every sample to the image that has been sampled from
    std::vector<uint32_t> bg_image_idx_ ; // used to address the background image to sample color values from
    std::vector<std::string> image_ids_ ; // ids (views) of each image (for debugging purposes)
    std::vector<std::string> labels_ ;    // label of each image
    std::vector<Eigen::Vector3f> boxes_ ; // object bounding boxes
    std::vector<Eigen::Vector3f> coordinates_ ; // 3D coordinates of each sample point
    std::vector<std::string> label_map_ ;
    std::vector<cvx::viz::ScenePtr> models_ ;
    std::vector<Eigen::Matrix4f> world_to_model_ ;
    std::vector<cvx::util::PointList3f> clouds_ ;

    void makeRandomSamples(const cv::Mat &rgb, const cv::Mat &depth, const PinholeCamera &cam,
                           RNG &g, uint n_samples, const uint32_t image_index, const std::string &label) ;
    void makeRandomSamples(const cv::Mat &depth, const cv::Mat &mask, RNG &g, uint n_samples) ;

    void makeBackgroundSamples(uint bg_image_idx, uint fg_image_idx, RNG &g, uint n_samples) ;

    float randomBackgroundDepth(sample_idx_t idx, uint16_t px, uint16_t py) const { return 0; }
    float randomBackgroundColor(uint16_t channel) const { return 0 ; }

};

#endif
