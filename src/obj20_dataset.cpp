#include "obj20_dataset.hpp"

#include <cvx/util/misc/path.hpp>
#include <cvx/util/misc/sequence.hpp>

#include <fstream>
#include <regex>

using namespace cvx::util ;
using namespace cvx::viz ;
using namespace std ;
using namespace Eigen ;

static std::istream& safe_getline(std::istream& is, std::string& t)
{
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for(;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if(sb->sgetc() == '\n')
                sb->sbumpc();
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if(t.empty())
                is.setstate(std::ios::eofbit);
            return is;
        default:
            t += (char)c;
        }
    }
}

void OBJ20_Dataset::load_pose(const string &info_path, string &label)
{
    ifstream strm(info_path.c_str()) ;

    string line ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;

    label = line ;

    safe_getline(strm, line) ;

    Affine3f a ;
    Matrix3f rotation ;
    strm >> rotation(0, 0) >> rotation(0, 1) >> rotation(0, 2) ;
    strm >> rotation(1, 0) >> rotation(1, 1) >> rotation(1, 2) ;
    strm >> rotation(2, 0) >> rotation(2, 1) >> rotation(2, 2) ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;
    Vector3f tr ;
    strm >> tr.x() >> tr.y() >> tr.z() ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;
    Vector3f extent ;
    strm >> extent.x() >> extent.y() >> extent.z() ;

    a.linear() = rotation ;
    a.translation() = tr ;

    Matrix4f axis_switch = Matrix4f::Identity() ;
    axis_switch(1, 1) = -1 ;
    axis_switch(2, 2) = -1 ;

    poses_.push_back(a.inverse().matrix() * axis_switch) ;
    boxes_.push_back(extent) ;

}


static void parse_label(const string &str, string &label, string &variant) {
    static regex r("[a-zA-Z0-9]+_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)$") ;

    smatch m ;
    if ( regex_search(str, m, r) ) {
        label = m[1] ;
        variant = m[2] ;
    }
}

void OBJ20_Dataset::loadImages(const std::string &image_path, const PinholeCamera &cam, uint num_images,
                         uint fg_samples_per_image, uint bg_samples_per_image)
{
    RNG g;

    uint  n_images = 0 ;

    DirectoryIterator it(image_path, DirectoryFilters::MatchDirectories), end ;

    FileSequence infoseq("info_", ".txt"), depthseq("depth_", ".png"), clrseq("color_", ".png"), maskseq("mask_", ".png") ;

  //  boost::uniform_int<> bg_image_rng(0, bg_rgb_images_.size()-1) ;

    for ( ; it != end ; ++it ) {

        string dir = it->path() ;

        Path ipath(image_path, dir) ;

        string label, variant ;

        parse_label(dir, label, variant) ;

        if ( variant == "spot" || variant == "dark" ) continue ;

        label_map_.push_back(label) ;

        vector<string> info_files = Path::glob( (ipath / "info").toString(), "info_*.txt", false ) ;

        num_images = std::min((uint)info_files.size(), num_images) ;

        std::vector<uint> v ;
        for(uint i=0 ; i<info_files.size() ; i++) v.push_back(i) ;

        g.shuffle(v) ;

        for(uint i=0 ; i<num_images ; i++)
        {
            const string &info_file_path = info_files[v[i]]  ;

            string clabel ;
            load_pose(info_files[v[i]], clabel) ;

            int id = infoseq.parseFrameId(info_file_path) ;

            cout << label << ' ' << id << endl ;

            Path depth_image_path = ipath / "depth_noseg" / depthseq.format(id) ;
            Path rgb_image_path = ipath / "rgb_noseg" / clrseq.format(id) ;
            Path mask_image_path = ipath / "mask" / maskseq.format(id) ;

            cv::Mat depth = cv::imread(depth_image_path.toString(), -1) ;
            cv::Mat rgb = cv::imread(rgb_image_path.toString()) ;
            cv::Mat mask = cv::imread(mask_image_path.toString()) ;
            cv::Mat plane = makePlanarDepth(cam, poses_.back(), boxes_.back()) ;

            depth_images_.push_back(depth) ;
            rgb_images_.push_back(rgb) ;
            masks_.push_back(mask) ;
            planar_images_.push_back(plane) ;

            if ( mask.type() == CV_8UC3 ) {
                cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY) ;
            }

            image_ids_.push_back(to_string(id)) ;

            makeRandomSamples(depth, mask, cam, g, fg_samples_per_image, n_images, label) ;
            makeBackgroundSamples(g.uniform<int>(0, bg_rgb_images_.size()-1), n_images, g, bg_samples_per_image) ;
            n_images ++ ;
        }

    }


}


void OBJ20_Dataset::loadBackgroundImages(const string &image_path, uint num_images)
{
    RNG g ;

    Path dir(image_path) ;

    vector<string> rgb_files = Path::glob(( dir / "rgb_noseg/").toString(),  "color_*.png", false) ;
    num_images = std::min((uint)rgb_files.size(), num_images) ;

    std::vector<uint> v ;
    for(uint i=0 ; i<rgb_files.size() ; i++) v.push_back(i) ;

    g.shuffle(v) ;

    FileSequence rgbseq("color_", ".png"), depthseq("depth_", ".png") ;

    for(uint i=0 ; i<num_images ; i++)
    {
        const string &rgb_image_path = rgb_files[i] ;
        int id = rgbseq.parseFrameId(rgb_image_path) ;

        Path depth_image_path = dir / "depth_noseg" / depthseq.format( id ) ;

        cv::Mat depth = cv::imread(depth_image_path.toString(), -1) ;
        cv::Mat rgb = cv::imread(rgb_image_path) ;

        bg_depth_images_.push_back(depth) ;
        bg_rgb_images_.push_back(rgb) ;
    }

}

bool OBJ20_Dataset::load_cloud(const string &fp, const Matrix4f &world_to_model, PointList3f &cloud) {

    ifstream strm(fp.c_str()) ;

    Isometry3f cw(world_to_model) ;

    while ( strm ) {
        float x, y, z ;
        strm >> x >> y >> z ;

        Vector3f pt(x, y, z) ;

        Vector3f q = cw * pt ;

        cloud.push_back(q) ;
    }

    return true ;
}


bool OBJ20_Dataset::load_camera(const string &dir, Matrix4f &cam)
{
    // we have to look for the first camera file

    bool found = false ;
    string info_file_path ;

    FileSequence infoseq("info_", ".txt") ;

    for( uint count = 0 ; count<100 ; count++ ) {
        Path p(dir, infoseq.format(count)) ;
        if ( p.exists() ) {
            found = true ;
            info_file_path = p.toString() ;
            break ;
        }
    }

    if ( !found ) return false ;

    ifstream strm(info_file_path) ;

    string line ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;

    safe_getline(strm, line) ;

    Affine3f a ;
    Matrix3f rotation ;
    strm >> rotation(0, 0) >> rotation(0, 1) >> rotation(0, 2) ;
    strm >> rotation(1, 0) >> rotation(1, 1) >> rotation(1, 2) ;
    strm >> rotation(2, 0) >> rotation(2, 1) >> rotation(2, 2) ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;
    Vector3f tr ;
    strm >> tr.x() >> tr.y() >> tr.z() ;
    safe_getline(strm, line) ;
    safe_getline(strm, line) ;
    Vector3f extent ;
    strm >> extent.x() >> extent.y() >> extent.z() ;


    a.linear() = rotation  ;
    a.translation() = tr ;

    cam = a.matrix().inverse() ;

    return true ;

}

void OBJ20_Dataset::loadModels(const string &models_path) {

    uint16_t n_labels = 0 ;

    DirectoryIterator it(models_path, DirectoryFilters::MatchDirectories), end ;

    for ( ; it != end ; ++ it ) {

        string dir = it->path() ;

        Path ipath(models_path, dir) ;

        string label, variant ;

        parse_label(dir, label, variant) ;

        if ( variant ==  "spot" || variant == "dark" ) continue ;

        cout << "loading: " << label << endl ;

        label_map_.push_back(label) ;

        Path model_path(ipath, "object.obj") ;
        Path cloud_path(ipath , "object.xyz") ;

        Matrix4f world_to_model ;
        load_camera( (ipath / "info").toString(), world_to_model ) ;

        world_to_model_.push_back(world_to_model) ;
        clouds_.push_back(PointList3f()) ;
        load_cloud(cloud_path.toString(), world_to_model, clouds_[n_labels] ) ;

 //               if ( label == "Kinfu_BattleCat1_light" ) {
        try {

            ScenePtr scene(new Scene) ;
           scene->load(model_path.toString()) ;
            models_.push_back( scene)  ;
  //          data.scene_ = Scene::load(model_path.native()) ;
//            data.renderer_ = boost::shared_ptr<SceneRenderer>(new SceneRenderer(data.scene_, rctx)) ;
        }
        catch (...) {
            cerr << "error loading model from:" << model_path << endl ;
            continue ;
        }

//             }

        n_labels ++ ;
    }


}
