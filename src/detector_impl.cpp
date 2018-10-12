#include "detector_impl.hpp"
#include "util.hpp"
#include "icp.hpp"

#include <cvx/util/geometry/kdtree.hpp>
#include <cvx/util/geometry/octree.hpp>
#include <cvx/util/misc/cv_helpers.hpp>
#include <cvx/util/imgproc/rgbd.hpp>

#include <cvx/viz/scene/scene.hpp>
#include <cvx/viz/scene/camera.hpp>
#include <cvx/viz/renderer/renderer.hpp>

#include <fstream>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace cvx::util ;
using namespace cvx::viz ;
using namespace std;
using namespace Eigen ;

//#define DEBUG

void compute_model_bounds(const PointList3f &cloud, ModelData &data) {

    Vector3f c(0, 0, 0) ;
    uint count = 0 ;

    Vector3f minp(  std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()),
            maxp( -std::numeric_limits<float>::max(),
                  -std::numeric_limits<float>::max(),
                  -std::numeric_limits<float>::max()) ;

    for ( uint i=0 ; i<cloud.size() ; i++ ) {

        const Vector3f &q = cloud[i] ;
        count ++ ;

        c += q ;

        minp.x() = std::min(minp.x(), q.x()) ;
        minp.y() = std::min(minp.y(), q.y()) ;
        minp.z() = std::min(minp.z(), q.z()) ;

        maxp.x() = std::max(maxp.x(), q.x()) ;
        maxp.y() = std::max(maxp.y(), q.y()) ;
        maxp.z() = std::max(maxp.z(), q.z()) ;
    }

    data.bmin_ = minp ;
    data.bmax_ = maxp ;

    data.center_ = c/count ;

    float max_distance = 0 ;

    for(uint i=0 ; i<cloud.size() ; i++ )
        max_distance = std::max( max_distance, (cloud[i] - data.center_).squaredNorm() ) ;

    data.diameter_ = 2 * sqrt(max_distance) ;

    data.search_.train(cloud) ;

}

std::istream& safe_getline(std::istream& is, std::string& t) ;

bool ObjectDetectorImpl::init(const string &rf_path, const Dataset &ds) {

    forest_.read(rf_path) ;

    labels_ = ds.label_map_ ;
    for( uint i=0 ; i<ds.models_.size() ; i++ ) {
        ModelData data ;

        data.scene_ = ds.models_[i] ;
        data.cloud_ = ds.clouds_[i] ;
        data.renderer_ = std::make_shared<Renderer>(data.scene_) ;
        data.renderer_->init() ;
        data.camera_ = ds.world_to_model_[i] ;
        compute_model_bounds(ds.clouds_[i], data) ;

        model_map_[labels_[i]] = data ;
    }

    return true ;
}


void ObjectDetectorImpl::compute_label_probabilities(const vector< vector<RandomForest::Node *> > &leafs, vector< map<string, float> > &probs) {
    uint nt = leafs.size() ;
    uint nsamples = leafs[0].size() ;

    probs.resize(nsamples) ;

    for( uint i=0 ; i<nsamples ; i++ ) {

        float bgprod = 1 ;
        for( uint j=0 ; j<nt ; j++) {
            RandomForest::Node *n = leafs[j][i] ;

            float bg ;
            bg = n->data_.n_background_ / (float) n->data_.sample_count_ ;
            bgprod *= bg ;
        }


        float lsum = 0 ;
        map<string, float> lprods ;

        for ( uint k=0 ; k<labels_.size() ; k++ ) {
            string clabel = labels_[k] ;
            float lprod = 1 ;
            for( uint j=0 ; j<nt ; j++) {
                RandomForest::Node *n = leafs[j][i] ;

                float pl ;
                if ( n->data_.classes_.count(clabel) == 0 ) pl = 0 ;
                else
                    pl =  n->data_.classes_[clabel] / (float) n->data_.sample_count_ ;

                lprod *= pl ;
            }
            lprods[clabel] = lprod ;
            lsum += lprod ;
        }


        for ( uint k=0 ; k<labels_.size() ; k++ ) {
            float p = lprods[labels_[k]]/( lsum + bgprod + 1.0e-8) ;

            if ( p > 0 )
                probs[i][labels_[k]] = p ;
        }
    }
}


void ObjectDetectorImpl::sample_pose_hypotheses(const std::string &clabel,
                                                const PinholeCamera &cam,
                                                const Dataset &ds,
                                                const LeafList &leafs,
                                                const LabelProbList &probs,
                                                vector<Matrix4f> &poses)
{
    //boost::timer::auto_cpu_timer t ;

    vector<float> weights ;
    PointList2f pts ;

    ModelData &md = model_map_[clabel] ;

    float object_diameter = md.diameter_ ;

    for(uint i=0 ; i<probs.size() ; i++ ) {

        float p = 0.0 ;
        LabelProbMapType::const_iterator it = probs[i].find(clabel) ;
        if ( it != probs[i].end() ) p = it->second ;

         weights.push_back(p) ;

        const cv::Point &pt = ds.samples_[i] ;
        pts.push_back(Vector2f(pt.x, pt.y)) ;
    }

    KDTree2 ksearch(pts) ;

    std::discrete_distribution<> pt_sampler(weights.begin(), weights.end()) ;
    std::uniform_int_distribution<> tree_sampler(0, leafs.size() - 1) ;

    const cv::Mat &dmap = ds.depth_images_[0] ;

    uint max_iterations = params_.max_ransac_iterations_ ;

    // iterate until we obtain enough hypotheses or reach the maximum number of trials

#pragma omp parallel for
    for ( uint iterations = 0 ; iterations < max_iterations ; iterations++ ) {

        if ( poses.size() < params_.max_hypotheses_ ) {

            // sample first point
            uint idx0 = pt_sampler(g_.generator()) ;
            uint t0 = tree_sampler(g_.generator()) ;

            const cv::Point &pt = ds.samples_[idx0] ;
            float z = dmap.at<ushort>(pt.y, pt.x)/1000.0 ;
            float search_radius = cam.fx() * object_diameter / z  ;

            // sample second point within radius around first point

            vector<uint> indexes ;
            vector<float> distances ;

            ksearch.withinRadius(pts[idx0], search_radius * search_radius, indexes, distances);

            if ( indexes.empty() ) continue ;

            vector<float> weights1 ;

            for( uint i=1 ; i<indexes.size() ; i++ ) {

                uint idx = indexes[i] ;
                float p = 0.0 ;
                LabelProbMapType::const_iterator it = probs[idx].find(clabel) ;
                if ( it != probs[idx].end() ) p = it->second ;

                weights1.push_back(p) ;
            }

            std::discrete_distribution<> pt_radius_sampler(weights1.begin(), weights1.end()) ;

            uint idx1 = indexes[pt_radius_sampler(g_.generator())+1] ;
            uint t1 = tree_sampler(g_.generator()) ;

            if ( idx0 == idx1 ) continue ;

            uint idx2 = indexes[pt_radius_sampler(g_.generator())+1] ;
            uint t2 = tree_sampler(g_.generator()) ;

            if ( idx0 == idx2 || idx1 == idx2 ) continue ;

            // Now we have 3 points on the image. we obtain the corresponding forest predictions and image back-projected 3D coordinates

            Vector3f o0, o1, o2 ;

            if ( leafs[t0][idx0]->data_.coordinates_.count(clabel) == 0 ) continue ;
            else o0 = leafs[t0][idx0]->data_.coordinates_[clabel] ;

            if ( leafs[t1][idx1]->data_.coordinates_.count(clabel) == 0 ) continue ;
            else o1 = leafs[t1][idx1]->data_.coordinates_[clabel] ;

            if ( leafs[t2][idx2]->data_.coordinates_.count(clabel) == 0 ) continue ;
            else o2 = leafs[t2][idx2]->data_.coordinates_[clabel] ;

            Vector3f p0 = back_project(dmap, cam, ds.samples_[idx0]) ; // these are in camera coordinate frame
            Vector3f p1 = back_project(dmap, cam, ds.samples_[idx1]) ;
            Vector3f p2 = back_project(dmap, cam, ds.samples_[idx2]) ;

            Matrix3Xf src(3, 3), dst(3, 3) ;

            Affine3f m2w(md.camera_.inverse().eval()) ;
            o0 = m2w * o0 ; o1 = m2w * o1 ; o2 = m2w * o2 ; // these are in world coordinate frame

            src.col(0) = o0 ; src.col(1) = o1 ; src.col(2) = o2 ;
            dst.col(0) = p0 ; dst.col(1) = p1 ; dst.col(2) = p2 ;

            // find alignment transformation

            Affine3f H = find_rigid(src, dst); // c2w

            // reject erroneous ones

            if ( ( H * o0 - p0).norm() > 0.05 * object_diameter ) continue ;
            if ( ( H * o1 - p1).norm() > 0.05 * object_diameter ) continue ;
            if ( ( H * o2 - p2).norm() > 0.05 * object_diameter ) continue ;
#ifdef DEBUG
            cv::Mat clr = ds.rgb_images_[0].clone() ;

            cv::circle(clr, ds.samples_[idx0], 3, cv::Scalar(255, 255, 0), 4) ;
            cv::circle(clr, ds.samples_[idx1], 3, cv::Scalar(255, 255, 0), 4) ;
            cv::circle(clr, ds.samples_[idx2], 3, cv::Scalar(255, 255, 0), 4) ;

            cv::imwrite("/tmp/samples.png", clr) ;

            cout << H.matrix() << endl ;
#endif
            // store accepted pose
#pragma omp critical
            poses.push_back(H.matrix()) ; // c2w
        }
    }
}

float ObjectDetectorImpl::compute_energy(const std::string &clabel, const Matrix4f &pose, const LeafList &leafs,
                                         const LabelProbList &probs,
                                         const PinholeCamera &cam, const Dataset &ds)
{
    ModelData &data = model_map_[clabel] ;

    cv::Mat_<ushort> depth = ds.depth_images_[0] ;

    cv::Mat_<ushort> zbuffer = render_mask(clabel, cam, pose) ;
    uint w = zbuffer.cols, h = zbuffer.rows ;
#ifdef DEBUG
   cv::imwrite("/tmp/zz.png", zbuffer) ;
#endif
    // depth component

    float ed = 0 ;
    uint ed_count = 0 ;

    for( uint i=0 ; i<h ; i++ )
        for( uint j=0 ; j<w ; j++ ) {
            ushort oval = zbuffer[i][j] ;
            ushort ival = depth[i][j] ;

            if ( oval == 0 || ival == 0 ) continue ;

            Vector3f ip = cam.backProject(j, i, ival/1000.0) ;
            Vector3f op = cam.backProject(j, i, oval/1000.0) ;

            float e = std::min((ip - op).norm(), params_.depth_error_threshold_)/params_.depth_error_threshold_ ;
            ed_count ++ ;
            ed += e ;
        }

    if ( ed_count == 0 ) ed = FLT_MAX ;
    else ed /= ed_count ;

    // object component (this differs a bit from the paper since we iterate over the pixels that are sampled from the image instead over all pixels in the mask)

    float eo = 0 ;
    uint eo_count = 0 ;

    uint n_trees = leafs.size() ;
    uint n_samples = ds.samples_.size() ;

    for( uint i=0 ; i<n_samples ; i++ ) {
        const cv::Point &pt = ds.samples_[i] ;

        ushort val = zbuffer[pt.y][pt.x] ;
        if ( val == 0 ) continue ;

        // here we have only samples that fall inside the mask

        for( uint j=0 ; j<n_trees ; j++ ) {
            RandomForest::Node *n = leafs[j][i] ;

            float p ;

            map<string, uint64_t>::const_iterator it = n->data_.classes_.find(clabel) ;
            if ( it == n->data_.classes_.end() ) p = 1.0e-5 ;
            else p = it->second / (float)n->data_.sample_count_ ;

            eo += -log(p) ;
            eo_count ++ ;
        }

    }

    if ( eo_count == 0 ) eo = FLT_MAX ;
    else eo /= eo_count ;

    // coordinate component (same as above)

    float ec = 0 ;
    uint ec_count = 0 ;

    float cet = params_.coord_error_threshold_ * data.diameter_ ;
    cet = cet * cet ;

    Matrix4f cami(data.camera_.inverse()) ;
    Affine3f m2w(cami) ;

    for( uint i=0 ; i<n_samples ; i++ ) {
        const cv::Point &pt = ds.samples_[i] ;

        ushort ival = depth[pt.y][pt.x] ;
        ushort oval = zbuffer[pt.y][pt.x] ;
        if ( ival == 0 || oval == 0 ) continue ;

        // exclude pixels with low class probability
        LabelProbMapType::const_iterator it = probs[i].find(clabel) ;
        if ( it == probs[i].end() || it->second < params_.class_membership_threshold_ )
            continue ;

        // coordinates in world frame based on predicted pose

        Vector3f ip = cam.backProject(pt.x, pt.y, ival/1000.0) ;
        ip = Isometry3f(pose.inverse().eval()) * ip ;

        for( uint j=0 ; j<n_trees ; j++ ) {
            RandomForest::Node *n = leafs[j][i] ;

            map<string, Vector3f>::const_iterator it = n->data_.coordinates_.find(clabel) ;
            Vector3f op = m2w * it->second ;

            ec += std::min((ip - op).squaredNorm(), cet)/cet ;
            ec_count ++ ;
        }
    }

    if ( ec_count < 100 ) return FLT_MAX ;

    if ( ec_count == 0 ) eo = FLT_MAX ;
    else ec /= ec_count ;

#ifdef DEBUG
    cv::imwrite("/tmp/zbuffer.png", depthViz(zbuffer)) ;
    cout << ed << ' ' << eo << ' ' << ec << endl ;
#endif

    return params_.lambda_ed_ * ed + params_.lambda_eo_ * eo + params_.lambda_ec_ * ec ;
}

typedef pair<float, uint16_t> PoseHypothesis ;

class Comparator {
public:
    bool operator () (const PoseHypothesis &h1, const PoseHypothesis &h2 ) {
        return h1.first >= h2.first ;
    }
};

typedef priority_queue<PoseHypothesis, vector<PoseHypothesis>, Comparator> PQueueType ;

void ObjectDetectorImpl::find_pose_candidates(const string &clabel, const PinholeCamera &cam, const Dataset &ds, const ObjectDetectorImpl::LeafList &leafs,
                                              const ObjectDetectorImpl::LabelProbList &probs, std::vector<Matrix4f> &poses, vector<float> &errors)
{


    std::vector<Matrix4f> candidates ;
    sample_pose_hypotheses(clabel, cam, ds, leafs, probs, candidates) ;

    PQueueType q ;

    for( uint k=0 ; k<candidates.size() ; k++ ) {

        float e = compute_energy(clabel, candidates[k], leafs, probs, cam, ds) ;

#ifdef DEBUG
        cout << e << endl ;
#endif
        q.push(PoseHypothesis(e, k)) ;
    }

    if ( q.empty() ) return ;

    // keep only best hypotheses

    float hthresh = params_.best_hypothesis_threshold_ * q.top().first ;

    while ( !q.empty() ) {
        PoseHypothesis p = q.top() ; q.pop() ;
        if ( p.first < hthresh ) {
            poses.push_back(candidates[p.second]) ;
            errors.push_back(p.first) ;
        }
        if ( poses.size() == params_.max_retain_candidates_ ) break ;
    }

}

cv::Mat ObjectDetectorImpl::render_mask(const string &clabel, const PinholeCamera &cam, const Matrix4f &pose) {

    // transform from source camera frame to object frame, then according to the pose to the target camera frame and finally switch axes

    ModelData &data = model_map_[clabel] ;

    // the opengl camera points along the negative Z axis so we have to switch coordinate systems
    Matrix4f axis_switch = Matrix4f::Identity() ;
    axis_switch(1, 1) = -1 ;
    axis_switch(2, 2) = -1 ;
//                              c2w
    Matrix4f cc = axis_switch * pose  ;

    CameraPtr pcam(new PerspectiveCamera(cam)) ;
    pcam->setViewTransform(cc);
    pcam->setBgColor({0, 0, 0, 1});

    data.renderer_->render(pcam) ;

    return data.renderer_->getDepth() ;
}

void ObjectDetectorImpl::refine_pose_candidates(const string &clabel, const PinholeCamera &cam, const Dataset &ds, const ObjectDetectorImpl::LeafList &leafs,
                                                const ObjectDetectorImpl::LabelProbList &probs, std::vector<Matrix4f> &poses, std::vector<float> &errors)
{

    ModelData &data = model_map_[clabel] ;

    cv::Mat_<ushort> depth = ds.depth_images_[0] ;

    for( uint iter = 0 ; iter < params_.max_refinement_steps_ ; iter ++ ) {

        bool no_more_refinement_needed = true ;

        for( uint k=0 ; k<poses.size() ; k++ ) {

            Isometry3f H(poses[k]) ;

            cv::Mat_<ushort> zbuffer = render_mask(clabel, cam, poses[k]) ;

            uint w = zbuffer.cols, h = zbuffer.rows ;

            uint n_trees = leafs.size() ;
            uint n_samples = ds.samples_.size() ;

            // try to find all sample pixels within the object mask that lead to a small coordinate re-projection error

            vector<Vector3f> ipts, opts ;

            Matrix4f cami(data.camera_.inverse());
             Affine3f m2w(cami) ;

            for( uint i=0 ; i<n_samples ; i++ ) {
                const cv::Point &pt = ds.samples_[i] ;

                ushort val = zbuffer[pt.y][pt.x] ;
                if ( val == 0 ) continue ;

                Vector3f ip = back_project(depth, cam, pt) ;

                // here we have only samples that fall inside the mask

                float min_e = FLT_MAX ;
                uint best_j ;
                Vector3f bop ;

                // find best tree and associated re-projection error

                for( uint j=0 ; j<n_trees ; j++ ) {
                    RandomForest::Node *n = leafs[j][i] ;

                    Vector3f op = m2w * n->data_.coordinates_[clabel] ;

                    float e = ( H * op - ip).norm() ;
                    if ( e < min_e ) {
                        min_e = e ;
                        best_j = j ;
                        bop = op ;
                    }
                }

                if ( min_e < params_.inlier_threshold_ ) {
                    ipts.push_back(ip) ;
                    opts.push_back(bop) ;
                }
            }

            uint n_inliers = ipts.size() ;

            if ( n_inliers < 3 ) continue ;

            Eigen::Map<Matrix3Xf> src((float *)opts.data(), 3, n_inliers)  ;
            Eigen::Map<Matrix3Xf> dst((float *)ipts.data(), 3, n_inliers)  ;

            // find alignment transformation

            Affine3f Hr = find_rigid(src, dst);

            float e = compute_energy(clabel, Hr.matrix(), leafs, probs, cam, ds) ;

            if ( e < errors[k] ) {
                no_more_refinement_needed = false ;
                errors[k] = e ;
                poses[k] = Hr.matrix() ;
            }


        }

        if ( no_more_refinement_needed ) break ;

    }
}

float ObjectDetectorImpl::refine_pose_icp(const string &clabel, const cv::Mat &dim, const PinholeCamera &cam, const Matrix4f &pose)
{
    Isometry3f ipose(pose.inverse().eval()) ;

    cv::Mat_<ushort> zbuffer = render_mask(clabel, cam, pose) ;
    uint w = zbuffer.cols, h = zbuffer.rows ;

    cv::Mat_<ushort> depth(dim) ;

    PointList3f ipts, rpts ;

    for ( uint i=0 ; i<h ; i++ ) {
        for(uint j=0 ; j<w ; j++ ) {

            ushort zval = zbuffer[i][j] ;

            if ( zval == 0 ) continue ;

            ushort ival = depth[i][j] ;

            if ( ival == 0 ) continue ;

            Vector3f ip = cam.backProject(j, i, ival/1000.0) ;

            ipts.push_back(ip) ;
        }
    }

    // resample cloud

    sampleCloudCenters(ipts, 0.01, rpts, {-10, -10, -10}, {10, 10, 10});

    ModelData &data = model_map_[clabel] ;

    ICPAligner::Parameters icp_params ;
    icp_params.inlier_distance_threshold_ = data.diameter_  ;

    ICPAligner icp(icp_params) ;

    uint n_inliers ;
    float error = icp.align(data.search_, data.cloud_, rpts, ipose, n_inliers) ;

    return error ;

}


void ObjectDetectorImpl::detectAll(const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const PinholeCamera &cam,
                                   vector<ObjectDetector::Result> &results) {

    // pass image through the forest

    Dataset ds ;

    ds.addSingleImage(rgb, depth, mask, params_.n_samples_);

    LeafList leafs ;

    forest_.apply_parallel(ds, leafs) ;

    // compute label probabilities (eq. 1)

    LabelProbList probs ;

    compute_label_probabilities(leafs, probs);
#ifdef DEBUG
    show_classes(ds, probs) ;
    show_coords(ds, leafs, 0) ;
    show_coords(ds, leafs, 1) ;
    show_coords(ds, leafs, 2) ;
#endif

    for( uint k=0 ; k<labels_.size() ; k++ ) {

        vector<Matrix4f> poses ;
        vector<float> errors ;

        string clabel = labels_[k];

        find_pose_candidates(clabel, cam, ds, leafs, probs, poses, errors) ;
        refine_pose_candidates(clabel, cam, ds, leafs, probs, poses, errors) ;

        cv::Mat cim = ds.rgb_images_[0].clone() ;
#ifdef DEBUG
        for( uint i=0 ; i<poses.size() && i < 4; i++ ) {
            if ( i==0 ) draw_bbox(cim, cam, clabel, poses[i], cv::Vec3b(0, 255, 0)) ;
            else draw_bbox(cim, cam, clabel, poses[i], cv::Vec3b(0, 255, 255)) ;
        }

        imwrite("/tmp/detections.png", cim) ;
#endif
        if ( poses.size() > 0 && errors[0] < params_.max_energy_threshold_ ) {
            Matrix4f best_pose = poses[0] ;
            float error = errors[0] ;

            ObjectDetector::Result res ;
            res.label_ = clabel ;
            res.error_ = error ;
            res.pose_ = best_pose ;
            results.push_back(res) ;
        }
    }

}

bool ObjectDetectorImpl::detect(const std::string &label, const cv::Mat &rgb, const cv::Mat &depth, const cv::Mat &mask, const PinholeCamera &cam,
                                Matrix4f &pose, float &error) {

    // pass image through the forest

    Dataset ds ;

    ds.addSingleImage(rgb, depth, mask, params_.n_samples_);

    LeafList leafs ;

    forest_.apply_parallel(ds, leafs) ;

    // compute label probabilities (eq. 1)

    LabelProbList probs ;

    compute_label_probabilities(leafs, probs);

    assert( std::find(labels_.begin(), labels_.end(), label) != labels_.end() ) ;

    vector<Matrix4f> poses ;
    vector<float> errors ;

    find_pose_candidates(label, cam, ds, leafs, probs, poses, errors) ;
    refine_pose_candidates(label, cam, ds, leafs, probs, poses, errors) ;

    if ( poses.size() > 0 && errors[0] > params_.max_energy_threshold_ ) {
        Matrix4f best_pose = poses[0] ;
        float best_error = errors[0] ;
#ifdef DEBUG

        cv::Mat cim = ds.rgb_images_[0].clone() ;
        draw_bbox(cim, cam, label, best_pose, cv::Vec3b(0, 255, 255)) ;
        imwrite("/tmp/detection.png", cim) ;
#endif

        pose = best_pose ;
        error = best_error ;
        return true ;

    }
    return false ;
}

void ObjectDetectorImpl::show_classes(const Dataset &ds, const LabelProbList &probs) {

    cv::Mat_<cv::Vec3b> res(480, 640) ;

    res = cv::Vec3b(0, 0, 0);

    cv::Vec3b def_clrs[6] = { cv::Vec3b( 255, 0, 0), cv::Vec3b( 0, 255, 0 ), cv::Vec3b( 0, 0, 255 ), cv::Vec3b(255, 255, 0), cv::Vec3b(255, 0, 255), cv::Vec3b(0, 255, 255) } ;

    map<string, cv::Vec3b> cmap ;

    for( uint i=0 ; i<labels_.size() ; i++ ) {
        if ( i < 6 ) cmap[labels_[i]] = def_clrs[i] ;
        else cmap[labels_[i]] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255) ;
    }

    for( uint i=0 ; i<ds.samples_.size() ; i++ ) {

        float max_votes = 0 ;
        string best_label ;

        map<string, float>::const_iterator it = probs[i].begin() ;

        for ( ; it != probs[i].end() ; ++it ) {
            string label = it->first ;
            float p = it->second ;

            if ( p > 0.9 && p > max_votes ) {
                max_votes = p ;
                best_label = label ;
            }
        }

        const cv::Point &p = ds.samples_[i] ;
        cv::Vec3b clr = ( best_label.empty() ) ? cv::Vec3b(0, 0, 0) : cmap[best_label] ;

        res[p.y][p.x] = clr ;

    }

    cv::imwrite("/tmp/labels.png", res) ;
}

void ObjectDetectorImpl::show_coords(const Dataset &ds, const LeafList &leafs, uint t) {

    cv::Mat_<cv::Vec3b> res(480, 640) ;

    res = cv::Vec3b(0, 0, 0);

    for( uint i=0 ; i<ds.samples_.size() ; i++ ) {

        RandomForest::Node *node = leafs[t][i] ;

        float max_votes = node->data_.n_background_ ;
        string best_label ;

        map<string, uint64_t>::const_iterator it = node->data_.classes_.begin() ;
        for( ; it != node->data_.classes_.end() ; ++it ) {
            const string &label = it->first ;
            uint64_t count = it->second ;

            if ( count > max_votes ) {
                max_votes = count ;
                best_label = label ;
            }
        }

        if ( best_label.empty() ) continue ;

        Vector3f coords = node->data_.coordinates_[best_label] ;

        ModelData &md = model_map_[best_label] ;

        Vector3f vm = coords ;
        Vector3f ext = 1.5*(md.bmax_ - md.bmin_) ;

        uint8_t vx = 255*(vm.x()/ext.x() + 0.5);
        uint8_t vy = 255*(vm.y()/ext.y() + 0.5);
        uint8_t vz = 255*(vm.z()/ext.z() + 0.5);

        const cv::Point &p = ds.samples_[i] ;
        res[p.y][p.x] = cv::Vec3b(vx, vy, vz) ;
    }

    imwritef(res, "/tmp/coords_%02d.png", t) ;

}


void ObjectDetectorImpl::draw_bbox(cv::Mat &img, const PinholeCamera &cam, const string &clabel, const Matrix4f &pose, const cv::Vec3b &clr)
{
    ModelData &data = model_map_[clabel] ;

    Vector3f box[8] ;

    Isometry3f wc(pose) ;

    box[0] = wc * Vector3f(data.bmin_.x(), data.bmin_.y(), data.bmin_.z()) ;
    box[1] = wc * Vector3f(data.bmin_.x(), data.bmax_.y(), data.bmin_.z()) ;
    box[2] = wc * Vector3f(data.bmax_.x(), data.bmax_.y(), data.bmin_.z()) ;
    box[3] = wc * Vector3f(data.bmax_.x(), data.bmin_.y(), data.bmin_.z()) ;

    box[4] = wc * Vector3f(data.bmin_.x(), data.bmin_.y(), data.bmax_.z()) ;
    box[5] = wc * Vector3f(data.bmin_.x(), data.bmax_.y(), data.bmax_.z()) ;
    box[6] = wc * Vector3f(data.bmax_.x(), data.bmax_.y(), data.bmax_.z()) ;
    box[7] = wc * Vector3f(data.bmax_.x(), data.bmin_.y(), data.bmax_.z()) ;

    cv::Point pts[8] ;

    for( uint i=0 ; i<8 ; i++ ) {
        cv::Point2d p = cam.project(cv::Point3d(box[i].x(), box[i].y(), box[i].z())) ;
        pts[i] = cv::Point(p.x, p.y) ;
    }

    cv::line(img, pts[0], pts[1], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[1], pts[2], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[2], pts[3], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[3], pts[0], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;

    cv::line(img, pts[4], pts[5], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[5], pts[6], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[6], pts[7], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[7], pts[4], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;

    cv::line(img, pts[0], pts[4], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[1], pts[5], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[3], pts[7], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
    cv::line(img, pts[2], pts[6], cv::Scalar(clr[0], clr[1], clr[2]), 2) ;
}


void ObjectDetectorImpl::draw_results(cv::Mat &img, const PinholeCamera &cam, const std::vector<ObjectDetector::Result> &results ) {
    cv::Vec3b def_clrs[6] = { cv::Vec3b( 255, 0, 0), cv::Vec3b( 0, 255, 0 ), cv::Vec3b( 0, 0, 255 ), cv::Vec3b(255, 255, 0), cv::Vec3b(255, 0, 255), cv::Vec3b(0, 255, 255) } ;

    map<string, cv::Vec3b> cmap ;

    for( uint i=0 ; i<labels_.size() ; i++ ) {
        if ( i < 6 ) cmap[labels_[i]] = def_clrs[i] ;
        else cmap[labels_[i]] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255) ;
    }

    for( uint i=0 ; i<results.size() ; i++ ) {
        const ObjectDetector::Result &res = results[i] ;
        cv::Vec3b clr = cmap[res.label_] ;
        draw_bbox(img, cam, res.label_, res.pose_, clr) ;
    }

}


void ObjectDetectorImpl::draw_result(cv::Mat &img, const PinholeCamera &cam, const std::string &clabel, const Matrix4f &pose, const cv::Vec3b &clr) {
    draw_bbox(img, cam, clabel, pose, clr) ;
}

void ObjectDetectorImpl::refine(const string &label, const cv::Mat &depth, const PinholeCamera &cam, Matrix4f &pose, float &error)
{
    error = refine_pose_icp(label, depth, cam, pose) ;
}
