#include "detector.hpp"
#include "certh_dataset.hpp"

using namespace cvx::util ;
using namespace std ;
using namespace Eigen ;

int main(int argc, char *argv[]) {

    PinholeCamera cam(570, 570, 640/2, 480/2, cv::Size(640, 480)) ;

//    cv::Mat rgb = cv::imread("/home/malasiot/tmp/obj_reconstruction/soft_scrub/rgb_00025.png") ;
//    cv::Mat depth = cv::imread("/home/malasiot/tmp/obj_reconstruction/soft_scrub/depth_00025.png", -1) ;

    cv::Mat rgb = cv::imread("/home/malasiot/tmp/test_orec/rgb_00007.png") ;
    cv::Mat depth = cv::imread("/home/malasiot/tmp/test_orec/depth_00007.png", -1) ;

    CERTH_Dataset ds ;
    ds.loadModels("/home/malasiot/tmp/obj_reconstruction/") ;

    ObjectDetector::Parameters params ;

    params.n_samples_ = 30000 ;
    params.max_ransac_iterations_ = 5000 ;
    ObjectDetector detector(params) ;
    detector.init("/home/malasiot/tmp/obj_reconstruction/", ds) ;
/*
    // find all planes in the image sorted by size
    vector<Vector4f> planes ;
    findAllPlanes(depth, cam, planes, 1000, 0.01, 3.0) ;

    cv::Mat mask = segmentPointsAbovePlane(depth, cam, planes[0]) ;

    cv::imwrite("/tmp/mask.png", mask) ;

    vector<ObjectDetector::Result> results ;
    detector.detectAll(rgb, depth, mask, cam, results) ;

    for(uint i=0 ; i<results.size() ; i++) {
        detector.refine(results[i].label_, depth, cam, results[i].pose_, results[i].error_) ;
    }
    detector.draw(rgb, cam, results) ;

    cv::imwrite("/tmp/result.png", rgb) ;
*/
    cout << "ok" << endl ;
}
