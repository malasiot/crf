#include "rf.hpp"
#include "certh_dataset.hpp"

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
    bool train_structure = false, train_retrofit = false ;

    ArgumentParser args ;
    args.setDescription("Usage: train [options] <data_dir> <out_dir>");
    args.addOption("-h|--help", show_help).setMaxArgs(0).setDescription("This help message").setImplicit("true") ;
    args.addOption("--cores", n_cores).setName("<n>").setDescription("number of cores to use") ;
    args.addOption("--bg-images", bg_image_folder).setName("<folder>").setDescription("Background image folder").required() ;
    args.addOption("--structure", train_structure).setMaxArgs(0).setDescription("train forest structure").setImplicit("true") ;
    args.addOption("--retrofit", train_retrofit).setMaxArgs(0).setDescription("populate forest leafs").setImplicit("true") ;
    args.addPositional(image_folder) ;
    args.addPositional(out_path) ;


        try {
            args.parse(argc, argv) ;

            if ( show_help )
                args.printUsage(std::cout) ;

        } catch ( ArgumentParserException &e ) {
            cout << e.what() << endl ;
            args.printUsage(std::cerr) ;
            return 1 ;
        }


    // parse arguments

    omp_set_num_threads(n_cores);

    PinholeCamera cam(550, 550, 640/2, 480/2, cv::Size(640, 480)) ;

    if ( train_structure ) {

        CERTH_Dataset ds ;
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

        CERTH_Dataset ds ;

        ds.loadBackgroundImages(bg_image_folder, 500);
        ds.loadImages(image_folder, cam, 100, 1500, 1500) ;

        RandomForest forest ;
        forest.read(out_path) ;

        TrainingParametersRegression rparams ;
        forest.retrofit(ds, cam, rparams);

        forest.write(out_path) ;
    }
}
