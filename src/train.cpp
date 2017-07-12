#include "rf.hpp"

#include <cvx/util/misc/arg_parser.hpp>

#include <string>
#include <stdio.h>
#include <fstream>

using namespace std ;
using namespace cvx::util ;

int main(int argc, const char *argv[])
{
    string  image_folder, out_path, bg_image_folder ;
    uint n_cores = omp_get_max_threads() ;
    bool show_help = false ;
    bool train_structure = true, train_retrofit = true ;

    ArgumentParser args ;
    args.addOption("-h|--help", show_help, true).setMaxArgs(0).setDescription("This help message") ;
    args.addOption("--cores", n_cores).setName("<n>").setDescription("number of cores to use") ;
    args.addOption("--bg-images", bg_image_folder).setName("<folder>").setDescription("Background image folder").required() ;
    args.addOption("--structure", train_structure, true).setMaxArgs(0).setDescription("train forest structure") ;
    args.addOption("--retrofit", train_retrofit, true).setMaxArgs(0).setDescription("populate forest leafs") ;
    args.addPositional(image_folder) ;
    args.addPositional(out_path) ;

    if ( !args.parse(argc, argv) || show_help ) {
        cerr << "Usage: train [options] <data_dir> <out_dir>\n";
        cerr << "Options:\n" ;
        args.printOptions(cerr) ;
        return 1;
    }

    // parse arguments

    omp_set_num_threads(n_cores);

    PinholeCamera cam(570, 570, 640/2, 480/2, cv::Size(640, 480)) ;

    if ( train_structure ) {

        Dataset ds ;
//"/home/malasiot/Downloads/bgimages/data/BG_Rooms/"
        ds.loadBackgroundImages(bg_image_folder, 500);
        ds.loadImages(image_folder, cam, 100, 500, 500) ;

        RandomForest forest ;

        TrainingParametersStructure sparams ;
        sparams.num_samples_per_tree = 1000 * 1000 ;

        forest.train(ds, sparams) ;
        forest.write(out_path) ;
    }

    if ( train_retrofit ) {

        Dataset ds ;

        ds.loadBackgroundImages(bg_image_folder, 500);
        ds.loadImages(image_folder, cam, 100, 1500, 1500) ;

        RandomForest forest ;
        forest.read(out_path) ;

        TrainingParametersRegression rparams ;
        forest.retrofit(ds, cam, rparams);

        forest.write(out_path) ;
    }
}
