#ifndef __ICP_HPP__
#define __ICP_HPP__

#include <Eigen/Geometry>
#include <vector>
#include <cvx/util/geometry/kdtree.hpp>

// simple ICP point-to-point alignment class

class ICPAligner {
public:

    struct Parameters {
        float inlier_distance_threshold_ ;
        float inlier_threshold_update_factor_ ;
        uint max_iterations_ ;
        uint min_inliers_ ;
        float epsilon_ ;     // difference in squared distance error to stop iterations

        Parameters():
            inlier_distance_threshold_(0.05),
            max_iterations_(10),
            epsilon_(1.0e-10),
            min_inliers_(3),
            inlier_threshold_update_factor_(3)
        {}

    } ;

    ICPAligner(const Parameters &params): params_(params) {}
    ICPAligner() {}

    // Point cloud alignment with ICP. Variable target is the static cloud and src is the moving cloud. Variable pose should be initialized
    // with the initial transform

    float align(const std::vector<Eigen::Vector3f> &target, const std::vector<Eigen::Vector3f> &src, Eigen::Isometry3f &pose, uint &n_inliers) {
        cvx::util::KDTree3 tree(target) ;
        return align(tree, target, src, pose, n_inliers) ;
    }

    // usefull when one needs to do several alignments to the same model
    float align(cvx::util::KDTree3 &stree, const std::vector<Eigen::Vector3f> &target, const std::vector<Eigen::Vector3f> &src, Eigen::Isometry3f &pose,
                uint &n_inliers) ;


private:

    Parameters params_ ;
};

#endif
