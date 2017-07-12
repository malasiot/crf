#ifndef __UTIL_H__
#define __UTIL_H__

#include <Eigen/Geometry>
#include <vector>
#include <boost/random.hpp>

#include <opencv2/opencv.hpp>
#include <cvx/util/camera/camera.hpp>
#include <cvx/util/geometry/kdtree.hpp>
#include <cvx/util/math/rng.hpp>

// perform mean shift clustering of the points and choose the mode with larger number of votes

void mean_shift(cvx::util::RNG &g, const std::vector<Eigen::Vector3f> &pts, Eigen::Vector3f &mode, float &weight, float sigma,
                    unsigned int maxIter, float seed_perc, uint min_seeds) ;

Eigen::Vector3f back_project(const cv::Mat &depth, const cvx::util::PinholeCamera &cam, const cv::Point &pt) ;

// Kabsch algorithm for rigid pose estimation between two point clouds ( finds T to minimize sum_i ||P_i * T - Q_i|| )

Eigen::Isometry3f find_rigid(const Eigen::Matrix3Xf &P, const Eigen::Matrix3Xf &Q) ;
Eigen::Isometry3f find_rigid(const std::vector<Eigen::Vector3f> &P, const std::vector<Eigen::Vector3f> &Q) ;

void save_cloud_obj(const std::string &file_name, const std::vector<Eigen::Vector3f> &cloud) ;

#endif
