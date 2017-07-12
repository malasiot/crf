#include "util.hpp"

#include <Eigen/Geometry>
#include <vector>
#include <map>
#include <fstream>

#include <float.h>


#include <cvx/util/math/rng.hpp>
#include <cvx/util/geometry/kdtree.hpp>

using namespace Eigen ;
using namespace std ;
using namespace cvx::util ;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static uint closest_center(const Vector3f &p, const vector<Vector3f> &c, float &closest_dist)
{
    float min_dist = FLT_MAX ;
    uint best_idx ;

    for(uint i=0 ; i<c.size() ; i++ )
    {
        float dist = (c[i] - p).norm() ;
        if ( dist < min_dist ) {
            min_dist = dist ;
            best_idx = i ;
        }
    }

    closest_dist = min_dist ;

    return best_idx ;

}

static void pruneModes(vector<Vector3f> &centers, vector<float> &cweights, float threshold, vector<uint> &cassign )
{
    uint n = centers.size() ;

    if ( n == 0 ) return ;

    vector<Vector3f> prunned ;
    vector<uint> idxs ;
    vector<float> cwassign ;

    idxs.resize(n) ;

    prunned.push_back(centers[0]) ;
    cassign.push_back(1) ;
    cwassign.push_back(cweights[0]) ;
    idxs[0] = 0 ;

    for( uint j=1 ; j<n ; j++ )
    {
        float closest_dist ;
        uint closest = closest_center(centers[j], prunned, closest_dist) ;

        if ( closest_dist < threshold )
        {
            assert(closest < cwassign.size()) ;

            cassign[closest] ++ ;
            cwassign[closest] += cweights[j] ;
            idxs[j] = closest ;
        }
        else {
            prunned.push_back(centers[j]) ;
            cassign.push_back(1) ;
            cwassign.push_back(cweights[j]) ;
            idxs[j] = cassign.size() - 1 ;
        }
    }

    centers = prunned ;
    cweights = cwassign ;
}

struct WeightSorter
{
    WeightSorter(const vector<float> &weights): weights_(weights) {}

    bool operator () (const int a, const int b) { return weights_[a] >= weights_[b] ; }
    const vector<float> weights_ ;
};



void mean_shift(RNG &g, const std::vector<Vector3f> &pts, Vector3f &mode,  float &weight, float sigma,
                    unsigned int maxIter, float seed_perc, uint min_seeds)
{
    uint n = pts.size() ;

    uint n_seed = std::max((uint)(n * seed_perc), std::min(n, min_seeds)) ;

    vector<uint> seed_idx ;

    for( uint i=0 ; i<n ; i++ )
        seed_idx.push_back(i) ;

    g.shuffle(seed_idx) ;

    Vector3f best_center ;
    float max_weight = -FLT_MAX ;

    for(uint r=0 ; r<n_seed ; r++)
    {
        const Vector3f &seed = pts[seed_idx[r]] ;

        Vector3f center = seed ;
        double center_dist ;
        uint iter = 0 ;
        float weight ;

        // shift center until convergence

        do {

            Vector3f new_center(0, 0, 0) ;
            double denom = 0.0 ;

            for(uint i=0 ; i<n ; i++ )
            {
                float sqd = (center - pts[i]).squaredNorm() ;
                double ep = exp(-sqd/ (2.0 * sigma * sigma));

                denom += ep ;

                new_center += ep * pts[i] ;
            }

            new_center /= denom ;

            center_dist = (new_center - center).norm() ;

            ++iter ;

            center = new_center ;

            weight = denom ;

        } while ( center_dist > 1.0e-7  && iter < maxIter ) ;

        if ( weight > max_weight ) {
            max_weight = weight ;
            best_center = center ;
        }
    }

    mode = best_center ;
    weight = max_weight ;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// rigid pose estimation

Isometry3f find_rigid(const Matrix3Xf &P, const Matrix3Xf &Q) {

    // Default output
    Isometry3f A;
    A.linear() = Matrix3f::Identity(3, 3);
    A.translation() = Vector3f::Zero();

    if (P.cols() != Q.cols())
        throw "Find3DAffineTransform(): input data mis-match";

    // Center the data
    Vector3f p = P.rowwise().mean();
    Vector3f q = Q.rowwise().mean();

    Matrix3Xf X = P.colwise() - p;
    Matrix3Xf Y = Q.colwise() - q;

    // SVD
    MatrixXf Cov = X*Y.transpose();
    JacobiSVD<MatrixXf> svd(Cov, ComputeThinU | ComputeThinV);

    // Find the rotation, and prevent reflections
    Matrix3f I = Matrix3f::Identity(3, 3);
    double d = (svd.matrixV()*svd.matrixU().transpose()).determinant();
    (d > 0.0) ? d = 1.0 : d = -1.0;
    I(2, 2) = d;

    Matrix3f R = svd.matrixV()*I*svd.matrixU().transpose();

    // The final transform
    A.linear() = R;
    A.translation() = q - R*p;

    return A;
}

Isometry3f find_rigid(const vector<Vector3f> &src, const vector<Vector3f> &dst) {

    assert( src.size() == dst.size() ) ;

    Eigen::Map<Matrix3Xf> m_src((float *)src.data(), 3, src.size())  ;
    Eigen::Map<Matrix3Xf> m_dst((float *)dst.data(), 3, dst.size())  ;

    return find_rigid(m_src, m_dst) ;
}

//////////////////////////////////////////////////////////////


Vector3f back_project(const cv::Mat &depth, const PinholeCamera &cam, const cv::Point &pt) {
    float z = depth.at<ushort>(pt)/1000.0 ;
    return cam.backProject(pt.x, pt.y, z) ;
}


void save_cloud_obj(const string &file_name, const std::vector<Vector3f> &cloud)
{
    ofstream strm(file_name.c_str()) ;

    for( uint i=0 ; i<cloud.size() ; i++ )
        strm << "v " << cloud[i].adjoint() << endl ;
}
