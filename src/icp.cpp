#include "icp.hpp"
#include "util.hpp"

#include <float.h>

using namespace Eigen ;
using namespace std ;
using namespace cvx::util;

float ICPAligner::align(KDTree3 &search, const vector<Vector3f> &target_pts, const std::vector<Vector3f> &src, Isometry3f &pose, uint &n_inliers)
{
    Isometry3f current(pose) ;

    float sq_distance_threshold = params_.inlier_distance_threshold_ * params_.inlier_distance_threshold_ ;

    float current_error = FLT_MAX, previous_error = FLT_MAX ;

    for( uint iter = 0 ; iter < params_.max_iterations_ ; ++iter ) {

        vector<Vector3f> inliers_src, inliers_dst, inliers_trans ;
        float sum_sq_distance = 0 ;

        for( uint i=0 ; i<src.size() ; i++ ) {
            const Vector3f &src_pt = src[i] ;
            Vector3f src_pt_trans = current * src_pt ;

            float dist ;
            uint idx = search.nearest(src_pt_trans, dist) ;

            if ( dist < sq_distance_threshold ) {
                inliers_src.push_back(src_pt) ;
                inliers_dst.push_back(target_pts[idx]) ;
                inliers_trans.push_back(src_pt_trans) ;
                sum_sq_distance += dist ;
            }
        }

        n_inliers = inliers_src.size() ;

        if ( n_inliers < params_.min_inliers_ ) break ;

        previous_error = current_error ;
        current_error = sum_sq_distance / n_inliers ;

        float delta = fabs(previous_error - current_error) ;
        if ( delta < params_.epsilon_ ) break ;

        sq_distance_threshold = params_.inlier_threshold_update_factor_ * current_error ;

        current = find_rigid(inliers_src, inliers_dst) ;
    }

    return current_error ;
}
