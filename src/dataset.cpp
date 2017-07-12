#include "dataset.hpp"
#include <fstream>
#include <cvx/util/math/rng.hpp>

using namespace cvx::util;
using namespace std ;
using namespace Eigen ;

void Dataset::makeRandomSamples(const cv::Mat &depth, const cv::Mat &omask, const PinholeCamera &cam, RNG &g, uint n_samples, const uint32_t image_index,
                                const string &label)
{
    uint w = depth.cols, h = depth.rows ;

    vector<uint> idx ;

    cv::Mat_<ushort> dim(depth) ;
    cv::Mat_<uchar> mask(omask) ;

    for(uint i=0 ; i<h ; i++)
        for(uint j=0 ; j<w ; j++)
        {
            if ( dim[i][j] == 0 ) continue ;
            if ( mask[i][j] == 0 ) continue ;

            idx.push_back(i * w + j) ;
        }

    uint nsamples = std::min(n_samples, (uint)idx.size()) ;

    g.shuffle(idx) ;

    // do the sampling

    for( uint i=0 ; i<nsamples ; i++ )
    {
        uint x = idx[i] % w ;
        uint y = idx[i] / w ;

        samples_.push_back(cv::Point(x, y)) ;
        image_idx_.push_back(image_index) ;

        ushort val = dim[y][x] ;

        cv::Point3d p = cam.backProject(cv::Point2d(x, y))  ;
        p *= val/1000.0 ;

        Vector3f gp = (poses_[image_index] * Vector4f(p.x, p.y, p.z, 1.0)).block<3, 1>(0, 0) ;

        coordinates_.push_back(gp) ;
        labels_.push_back(label) ;
        bg_image_idx_.push_back(-1) ;

    }

}

void Dataset::makeRandomSamples(const cv::Mat &depth, const cv::Mat &mask, RNG &g, uint n_samples)
{
    uint w = depth.cols, h = depth.rows ;

    vector<uint> idx ;

    cv::Mat_<ushort> dim(depth) ;
    cv::Mat_<uchar> roi(mask) ;

    for(uint i=0 ; i<h ; i++)
        for(uint j=0 ; j<w ; j++)
        {
            if ( dim[i][j] != 0 && roi[i][j] != 0 )
                idx.push_back(i * w + j) ;
        }

    uint nsamples = std::min(n_samples, (uint)idx.size()) ;

    g.shuffle(idx) ;

    // do the sampling

    for( uint i=0 ; i<nsamples ; i++ )
    {
        uint x = idx[i] % w ;
        uint y = idx[i] / w ;

        samples_.push_back(cv::Point(x, y)) ;
        image_idx_.push_back(0) ;
        labels_.push_back("test") ;
    }

}

void Dataset::makeBackgroundSamples(uint bg_idx, uint fg_idx, RNG &g, uint n_samples)
{
    cv::Mat_<ushort> depth(bg_depth_images_[bg_idx]) ;

    uint w = depth.cols, h = depth.rows ;

    for( uint i=0 ; i<n_samples ; i++ )
    {
        int x, y ;
        while (1) {
            x = g.uniform<int>(0, w-1) ;
            y = g.uniform<int>(0, h-1) ;
            if ( depth[y][x] != 0 ) break ;
        }

        samples_.push_back(cv::Point(x, y)) ;
        image_idx_.push_back(fg_idx) ;
        bg_image_idx_.push_back(bg_idx) ;
        coordinates_.push_back(Vector3f(0, 0, 0)) ;
        labels_.push_back(string()) ;
    }

}

Dataset::Dataset()
{

}



cv::Mat Dataset::makePlanarDepth(const PinholeCamera &cam, const Matrix4f &pose, const Vector3f &box)
{
    cv::Mat_<ushort> res(cam.sz()) ;

    float r0 = pose(1, 0), r1 = pose(1, 1), r2 = pose(1, 2), r3 = pose(1, 3) ;
    float nm = -box.y()/2.0 - r3 ;

    uint w = res.cols, h = res.rows ;
    for( uint y=0 ; y<h ; y++) {
        for( uint x=0 ; x<w ; x++ ) {
            float dm = r0 * x/cam.fx() + r1 * y/cam.fy() + r2 ;
            float val = nm * 1000.0/dm ;

            res[y][x] = (ushort) val ;
        }
    }

    return res ;

}



void Dataset::addSingleImage(const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, uint samples)
{
    RNG g;

    rgb_images_.push_back(rgb) ;
    depth_images_.push_back(depth) ;

    makeRandomSamples(depth, mask, g, samples) ;
}
