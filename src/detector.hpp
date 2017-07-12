#ifndef __OBJECT_DETECTOR_HPP__
#define __OBJECT_DETECTOR_HPP__

#include <cvx/util/camera/camera.hpp>

#include "dataset.hpp"

class ObjectDetectorImpl ;

class ObjectDetector {
public:

    struct Parameters {
        uint n_samples_ = 10000 ;                   // number of pixel sample to feed to random forest
        uint max_hypotheses_ = 250 ;                // maximum hypotheses to generate per class
        uint max_ransac_iterations_ = 10000 ;       // maximum iterations of sampling triplets
        float depth_error_threshold_ = 0.01 ;       // threshold used in eq. 3 to saturate large depth map incosistency errors
        float class_membership_threshold_ = 0.1 ;   // used in eq. 5 to exclude pixels that are unlikely to belong to the target class
        float coord_error_threshold_ = 0.2 ;        // threshold used in eq. 5 to saturate large coordinate prediction incosistency errors
                                                    // given as percentage of object diameter
        float lambda_ed_ = 1.5, lambda_eo_ = 1.0, lambda_ec_ = 1.0 ;
                                                    // weights used for the total energy (eq. 2)
        float max_retain_candidates_ = 25 ;         // maxiumum pose candidates to keep for refinement
        float best_hypothesis_threshold_ = 1.2 ;    // this is multiplied to the energy of the best pose hypthesis and used as a threshold to remove poses with high energy
        float inlier_threshold_ = 0.05 ;            // used to find inlier during the refinement step
        uint max_refinement_steps_ = 10 ;           // number of refinement iterations during which the energy decreases
        float max_energy_threshold_ = 6.0 ;

        float icp_align_threshold_ = 1.0e-5 ;        // validation step ICP error threshold (average L2 distance of inliers) to reject false positives
        float icp_inlier_percentage_ = 0.7 ;         // threshold on the percentage of image points that are detected as inliers after ICP iterations
    };

    ObjectDetector(const Parameters &params) ;

    bool init(const std::string &rf_path, const Dataset &model_data) ;

    struct Result {
        std::string label_ ;
        float error_ ;
        Eigen::Matrix4f pose_ ;
    };

    void detectAll(const cv::Mat &rgb, const cv::Mat &depth, const cvx::util::PinholeCamera &cam, std::vector<Result> &results) ;
    void detectAll(const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const cvx::util::PinholeCamera &cam, std::vector<Result> &results) ;

    bool detect(const std::string &label, const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const cvx::util::PinholeCamera &cam,
                Eigen::Matrix4f &pose, float &error) ;

    bool detect(const std::string &label, const cv::Mat &rgb, const cv::Mat &depth, const cvx::util::PinholeCamera &cam,
                Eigen::Matrix4f &pose, float &error) ;

    void refine(const std::string &label, const cv::Mat &depth, const cvx::util::PinholeCamera &cam,
                Eigen::Matrix4f &pose, float &error) ;

    void draw(cv::Mat &img, const cvx::util::PinholeCamera &cam, const std::vector<Result> &results ) ;
    void draw(cv::Mat &img, const cvx::util::PinholeCamera &cam, const std::string &clabel, const Eigen::Matrix4f &pose, const cv::Vec3b &clr) ;

private:

    std::shared_ptr<ObjectDetectorImpl> impl_ ;
    Parameters params_ ;
};


#endif
