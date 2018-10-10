#include "detector.hpp"
#include "certh_dataset.hpp"

#include <cvx/viz/gui/offscreen.hpp>

using namespace cvx::util ;
using namespace cvx::viz ;
using namespace std ;
using namespace Eigen ;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


int main(int argc, char *argv[]) {



    PinholeCamera cam(550, 550, 640/2, 480/2, cv::Size(640, 480)) ;

    OffscreenRenderingWindow ow(cam.sz().width, cam.sz().height) ;

//    cv::Mat rgb = cv::imread("/home/malasiot/tmp/obj_reconstruction/soft_scrub/rgb_00025.png") ;
//    cv::Mat depth = cv::imread("/home/malasiot/tmp/obj_reconstruction/soft_scrub/depth_00025.png", -1) ;

    cv::Mat rgb = cv::imread("/home/malasiot/tmp/crf/bunny/rgb_00142.png") ;
    cv::Mat depth = cv::imread("/home/malasiot/tmp/crf/bunny/depth_00142.png", -1) ;
    cv::Mat mask = cv::imread("/home/malasiot/tmp/crf/bunny/mask_00142.png", -1) ;


    CERTH_Dataset ds ;
    ds.loadModels("/home/malasiot/tmp/crf/") ;

    ObjectDetector::Parameters params ;

    params.n_samples_ = 1000 ;
    params.max_ransac_iterations_ = 1000 ;
    ObjectDetector detector(params) ;
    detector.init("/home/malasiot/tmp/crf/", ds) ;

    vector<ObjectDetector::Result> results ;
    detector.detectAll(rgb, depth, mask, cam, results) ;
/*
    for(uint i=0 ; i<results.size() ; i++) {
        cout << results[i].pose_.inverse() << endl ;
        detector.refine(results[i].label_, depth, cam, results[i].pose_, results[i].error_) ;
    }
*/
    detector.draw(rgb, cam, results) ;

    cv::imwrite("/tmp/result.png", rgb) ;
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
