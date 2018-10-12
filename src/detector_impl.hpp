#ifndef __OBJECT_DETECTOR_IMPL_HPP__
#define __OBJECT_DETECTOR_IMPL_HPP__

#include <map>
#include <string>

#include <cvx/viz/scene/scene.hpp>
#include <cvx/viz/renderer/renderer.hpp>
#include <cvx/util/camera/camera.hpp>
#include <cvx/util/geometry/kdtree.hpp>
#include <cvx/util/misc/path.hpp>

#include <Eigen/Core>

#include "rf.hpp"
#include "detector.hpp"

typedef std::map<std::string, uint16_t> LabelMapType ;
typedef std::map<std::string, float> LabelProbMapType ;

using RendererPtr = std::shared_ptr<cvx::viz::Renderer> ;
using ScenePtr = cvx::viz::ScenePtr ;

struct ModelData {
    ScenePtr scene_ ;
    cvx::util::PointList3f cloud_ ;
    cvx::util::KDTree3 search_ ;
    Eigen::Vector3f center_, bmin_, bmax_ ;
    float diameter_ ;
    RendererPtr renderer_ ;
    Eigen::Matrix4f camera_ ;
};

class ObjectDetectorImpl {

public:

    ObjectDetectorImpl(const ObjectDetector::Parameters &params): params_(params) {}

    bool init(const std::string &rf_path, const Dataset &models) ;

    void detectAll(const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const cvx::util::PinholeCamera &cam,
                   std::vector<ObjectDetector::Result> &results) ;
    bool detect(const std::string &label, const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const cvx::util::PinholeCamera &cam,
                   Eigen::Matrix4f &pose, float &error) ;

    void draw_results(cv::Mat &img, const cvx::util::PinholeCamera &cam, const std::vector<ObjectDetector::Result> &results ) ;
    void draw_result(cv::Mat &img, const cvx::util::PinholeCamera &cam, const std::string &clabel, const Eigen::Matrix4f &pose, const cv::Vec3b &clr) ;

    void refine(const std::string &label, const cv::Mat &depth, const cvx::util::PinholeCamera &cam, Eigen::Matrix4f &pose, float &error) ;

protected:

    typedef std::vector< std::vector<RandomForest::Node *> > LeafList ;
    typedef std::vector< std::map<std::string, float> > LabelProbList ;

    void compute_label_probabilities(const LeafList &leafs, LabelProbList &probs) ;

    void show_classes(const Dataset &ds, const LabelProbList &probs) ;
    void show_coords(const Dataset &ds, const LeafList &leafs, uint t) ;

    void sample_pose_hypotheses(const std::string &clabel, const cvx::util::PinholeCamera &cam,
                                const Dataset &ds,
                                const LeafList &leafs,
                                const LabelProbList &probs,
                                std::vector<Eigen::Matrix4f> &poses) ;

    bool load_cloud(const std::string &fp, ModelData &data) ;
    bool load_camera(const std::string &dir, Eigen::Matrix4f &cam) ;

    float compute_energy(const std::string &clabel, const Eigen::Matrix4f &poses, const LeafList &leafs,
                        const LabelProbList &probs,
                        const cvx::util::PinholeCamera &cam, const Dataset &ds) ;

    void draw_bbox(cv::Mat &img, const cvx::util::PinholeCamera &cam, const std::string &clabel, const Eigen::Matrix4f &pose, const cv::Vec3b &clr) ;


    void find_pose_candidates(const std::string &clabel, const cvx::util::PinholeCamera &cam,
                              const Dataset &ds,
                              const LeafList &leafs,
                              const LabelProbList &probs,
                              std::vector<Eigen::Matrix4f> &poses,
                              std::vector<float> &errors) ;

    void refine_pose_candidates(const std::string &clabel, const cvx::util::PinholeCamera &cam,
                                const Dataset &ds,
                                const LeafList &leafs,
                                const LabelProbList &probs,
                                std::vector<Eigen::Matrix4f> &poses,
                                std::vector<float> &errors) ;

    cv::Mat render_mask(const std::string &clabel, const cvx::util::PinholeCamera &cam, const Eigen::Matrix4f &pose) ;
    float refine_pose_icp(const std::string &clabel, const cv::Mat &dim, const cvx::util::PinholeCamera &cam, const Eigen::Matrix4f &pose) ;

    RandomForest forest_ ;
    std::vector<std::string> labels_ ;
    std::map<std::string, ModelData> model_map_ ;
    ObjectDetector::Parameters params_ ;
    RNG g_ ;
};


#endif
