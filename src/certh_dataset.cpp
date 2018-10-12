#include "certh_dataset.hpp"

#include <cvx/util/misc/path.hpp>
#include <cvx/util/misc/sequence.hpp>
#include <cvx/util/misc/strings.hpp>

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

void CERTH_Dataset::load_pose(const string &info_path, Matrix4f &pose)
{
    ifstream strm(info_path.c_str()) ;

    strm >> pose(0, 0) >> pose(0, 1) >> pose(0, 2) >> pose(0, 3);
    strm >> pose(1, 0) >> pose(1, 1) >> pose(1, 2) >> pose(1, 3);
    strm >> pose(2, 0) >> pose(2, 1) >> pose(2, 2) >> pose(2, 3);
    strm >> pose(3, 0) >> pose(3, 1) >> pose(3, 2) >> pose(3, 3);

}


static void parse_label(const string &str, string &label, string &variant) {
    static regex r("([a-zA-Z0-9]+)(?:_([a-zA-Z0-9]+))?$") ;

    smatch m ;
    if ( regex_search(str, m, r) ) {
        label = m[1] ;
        variant = m[2] ;
    }
}

static void parse_box(const string &dir, Vector3f &box) {
     ifstream strm( Path(dir, "box.txt").toString()) ;
     strm >> box.x() >> box.y() >> box.z() ;
}

void CERTH_Dataset::loadImages(const std::string &image_path, const PinholeCamera &cam, uint num_images,
                         uint fg_samples_per_image, uint bg_samples_per_image)
{
    RNG g;

    uint  n_images = 0 ;

    DirectoryIterator it(image_path, DirectoryFilters::MatchDirectories), end ;

    FileSequence poseseq("pose_", ".txt"), depthseq("depth_", ".png"), clrseq("rgb_", ".png"), maskseq("mask_", ".png") ;

    for ( ; it != end ; ++it ) {

        string dir = it->path() ;

        Path ipath(image_path, dir) ;

        if ( ( ipath / "ignore.txt" ).exists() ) continue ;

        string label, variant ;

        parse_label(dir, label, variant) ;

        label_map_.push_back(label) ;

        Vector3f box ;
        parse_box(ipath.toString(), box) ;

        Matrix4f w2m ;
        load_pose((ipath / "w2m.txt").toString(), w2m) ;

        vector<string> pose_files = Path::glob( ipath.toString(), "pose_*.txt", false ) ;
        num_images = std::min((uint)pose_files.size(), num_images) ;

        g.shuffle(pose_files) ;

        for(uint i=0 ; i<num_images ; i++)
        {
            const string &pose_file_path = pose_files[i]  ;

            Matrix4f pose ;
            load_pose(pose_file_path, pose) ;

            // we need camera-to-model
            poses_.push_back(w2m * pose) ;
            boxes_.push_back(box) ;

            int id = poseseq.parseFrameId(pose_file_path) ;

            cout << label << ' ' << id << endl ;

            Path depth_image_path = ipath / depthseq.format(id) ;
            Path rgb_image_path = ipath / clrseq.format(id) ;
            Path mask_image_path = ipath / maskseq.format(id) ;

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


void CERTH_Dataset::loadBackgroundImages(const string &image_path, uint num_images)
{
    RNG g ;

    Path dir(image_path) ;

    vector<string> rgb_files = Path::glob(( dir / "rgb_noseg/").toString(),  "color_*.png", false) ;
    num_images = std::min((uint)rgb_files.size(), num_images) ;

    g.shuffle(rgb_files) ;

    FileSequence rgbseq("color_", ".png"), depthseq("depth_", ".png") ;

    for ( uint i=0 ; i<num_images ; i++)
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

bool CERTH_Dataset::load_cloud(const string &fp, PointList3f &cloud) {

    ifstream strm(fp.c_str()) ;

    bool is_data = false ;
    uint n_vertices = 0 ;

    while (strm) {
        string line ;
        safe_getline(strm, line) ;

        if ( !is_data ) {
            vector<string> tokens = split(line, " \t");

            if ( tokens.size() == 1 && tokens[0] == "end_header") {
                is_data = true ;
            }
            else if ( tokens.size() == 3 && tokens[0] == "element" && tokens[1] == "vertex" ) {
                n_vertices = atoi(tokens[2].c_str() ) ;
            }
        }
        else {
            if ( cloud.size() == n_vertices ) break ;

            istringstream sl(line) ;
            float x, y, z ;
            sl >> x >> y >> z ;

            cloud.push_back(Vector3f(x, y, z)) ;

        }

    }

    return true ;
}

void CERTH_Dataset::loadModels(const string &models_path) {

    uint16_t n_labels = 0 ;

    DirectoryIterator it(models_path, DirectoryFilters::MatchDirectories), end ;

    for ( ; it != end ; ++ it ) {

        string dir = it->path() ;

        Path ipath(models_path, dir) ;

        if ( ( ipath / "ignore.txt" ).exists() ) continue ;

        string label, variant ;

        parse_label(dir, label, variant) ;

        cout << "loading: " << label << endl ;

        label_map_.push_back(label) ;

        Path model_path(ipath, "model.obj") ;
        Path cloud_path(ipath , "merged.ply") ;

        Matrix4f world_to_model ;
        load_pose( (ipath / "w2m.txt").toString(), world_to_model ) ;

        // poses already transform from camera to model
    //    world_to_model_.push_back(Matrix4f::Identity()) ;
         world_to_model_.push_back(world_to_model) ;
        clouds_.push_back(PointList3f()) ;
        load_cloud(cloud_path.toString(), clouds_.back() ) ;

        try {
            ScenePtr scene(new Scene) ;
            scene->load(model_path.toString()) ;
            models_.push_back( scene ) ;

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


