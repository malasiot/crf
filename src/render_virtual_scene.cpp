#include <cvx/viz/renderer/renderer.hpp>
#include <cvx/viz/scene/light.hpp>
#include <cvx/viz/scene/camera.hpp>
#include <cvx/viz/scene/material.hpp>
#include <cvx/util/geometry/viewpoint_sampler.hpp>
#include <cvx/util/misc/path.hpp>
#include <cvx/util/misc/strings.hpp>
#include <cvx/util/imgproc/rgbd.hpp>

#include <fstream>

using namespace cvx::viz ;
using namespace cvx::util ;

using namespace Eigen ;
using namespace std ;

void saveCloud(const std::string &filename, PointList3f &cloud, const Eigen::Matrix4f &mat) {
    ofstream strm(filename)     ;
    for( const Vector3f &v: cloud ) {
        strm << "v " << (mat * Vector4f(v.x(), v.y(), v.z(), 1)).adjoint() <<  endl ;
    }
}

int main(int argc, char *argv[]) {

    std::vector<Eigen::Matrix4f> views ;

    ViewPointSampler sampler ;
    sampler.setRadius(0.5, 1.2, 0.1);
    sampler.generate(200, views) ;

    ScenePtr s(new Scene) ;
    s->load("/home/malasiot/Downloads/bunny.obj") ;

    DirectionalLight *light = new DirectionalLight({0, 0, 1}) ;
    light->diffuse_color_ = {0.5, 0.5, 0.5} ;
    s->addLight(LightPtr(light)) ;

    OffscreenRenderingWindow w(640, 480) ;

    Renderer rdr(s) ;
    rdr.init() ;

    PinholeCamera camera(550, 550, 640/2.0, 480/2.0, cv::Size(640, 480)) ;

    sampler.exportCameras("/tmp/cameras.dae", views, camera) ;

    CameraPtr cam(new PerspectiveCamera(camera)) ;
    cam->setBgColor({0, 0.5, 1, 0}) ;

    string out_path("/home/malasiot/tmp/crf/bunny/") ;
    Path::mkdirs(out_path) ;

    Matrix4f axis_switch = Matrix4f::Identity() ;
    axis_switch(1, 1) = -1 ;
    axis_switch(2, 2) = -1 ;

    for( uint i=0 ; i<views.size() ; i++ ) {

        Matrix4f a = views[i] ;

//        cout << i << endl << a << endl ;

        cam->setViewTransform(a) ;
        rdr.render(cam) ;

        cv::Mat rgba = rdr.getColor(), rgb ;
        cv::Mat depth = rdr.getDepth(), mask ;

        vector<cv::Mat> channels;
        cv::split(rgba, channels);
        mask = channels[3] ;
        cv::Mat rgbc[] = { channels[0], channels[1], channels[2] } ;
        cv::merge(rgbc, 3, rgb) ;

        string rgb_filename = format("rgb_%05d.png", i) ;
        string depth_filename = format("depth_%05d.png", i) ;
        string pose_filename = format("pose_%05d.txt", i) ;
        string mask_filename = format("mask_%05d.png", i) ;
        string cloud_filename = format("cloud_%05d.obj", i) ;


        cv::imwrite( Path(out_path, rgb_filename).toString(), rgb) ;
        cv::imwrite( Path(out_path, depth_filename).toString(), depth) ;
        cv::imwrite( Path(out_path, mask_filename).toString(), mask) ;

        ofstream strm( Path(out_path,  pose_filename).toString().c_str()) ;
        strm << a.inverse() * axis_switch ; // this is needed to go from GL coordinate system to right handed coordinate system
/*
        EPointList3f cloud ;

        cv::Mat_<ushort> depth_(depth) ;

        for(int i=0 ; i<depth.rows ; i++)
            for(int j=0 ; j<depth.cols ; j++)
            {
                ushort val = depth_[i][j] ;

                if ( val == 0 ) continue ;

                Vector3f p = camera.backProject(j, i, val/1000.0) ;
                Vector3f q(p.x(), p.y(), p.z()) ;
                cloud.push_back(q) ;
            }

        saveCloud( Path(out_path, cloud_filename).toString(), cloud, a.inverse() * axis_switch) ;
*/

    }
}
