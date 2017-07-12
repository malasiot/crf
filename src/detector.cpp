#include "detector.hpp"
#include "detector_impl.hpp"

using namespace std ;

ObjectDetector::ObjectDetector(const Parameters &params): params_(params), impl_(new ObjectDetectorImpl(params)) {}

bool ObjectDetector::init(const string &rf_path, const Dataset &models) {
    return impl_->init(rf_path, models) ;
}

void ObjectDetector::detectAll(const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const cvx::util::PinholeCamera &cam, vector<Result> &results) {
    impl_->detectAll(rgb, depth, mask, cam, results) ;
}

void ObjectDetector::detectAll(const cv::Mat &rgb, const cv::Mat &depth, const cvx::util::PinholeCamera &cam, vector<Result> &results) {
    cv::Mat mask(rgb.size(), CV_8UC1, cv::Scalar(255)) ;
    impl_->detectAll(rgb, depth, mask, cam, results) ;
}

bool ObjectDetector::detect(const string &label, const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const cvx::util::PinholeCamera &cam,
                            Eigen::Matrix4f &pose, float &error) {
    return impl_->detect(label, rgb, depth, mask, cam, pose, error) ;
}

bool ObjectDetector::detect(const string &label, const cv::Mat &rgb, const cv::Mat &depth, const cvx::util::PinholeCamera &cam,
                            Eigen::Matrix4f &pose, float &error) {
    cv::Mat mask(rgb.size(), CV_8UC1, cv::Scalar(255)) ;
    return impl_->detect(label, rgb, depth, mask, cam, pose, error) ;
}

void ObjectDetector::refine(const string &label, const cv::Mat &depth, const cvx::util::PinholeCamera &cam, Eigen::Matrix4f &pose, float &error)
{
    impl_->refine(label, depth, cam, pose, error) ;
}

void ObjectDetector::draw(cv::Mat &img, const cvx::util::PinholeCamera &cam, const std::vector<ObjectDetector::Result> &results)
{
    impl_->draw_results(img, cam, results) ;
}
