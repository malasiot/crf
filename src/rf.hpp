#ifndef __RF_HPP__
#define __RF_HPP__

#include "dataset.hpp"

#include <vector>
#include <cvx/util/misc/binary_stream.hpp>
#include <cvx/util/math/rng.hpp>
#include <cvx/util/camera/camera.hpp>

/*
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/unordered_map.hpp>
*/

#include <algorithm>

#include <Eigen/Geometry>

struct TrainingParametersStructure {
    unsigned int max_depth = 64;                // maximum depth of each tree
    unsigned int num_random_samples = 1000 ;    // number of random features to test for each node
    double gain_threshold = -0.00001 ;          // threshold of gain achieved afetr splitting. if less than threshold node will be split.
    unsigned int num_trees = 3 ;                // number of trees in the forest
    unsigned int min_pts_per_leaf = 50 ;        // minimum number of data points falling on each leaf node
    unsigned int num_samples_per_tree = 5000*1000 ; // number of samples randomly selected from the original dataset to pass to each tree (with replacement)
    unsigned int num_thresholds_per_sample = 20 ;  // number of threshold to test for each feature

    unsigned int min_probe_offset_depth = 5, min_probe_offset_rgb = 0 ;
    unsigned int max_probe_offset_depth = 50, max_probe_offset_rgb = 20 ; // the offset around the current pixel to look at while computing features
    float depth_to_color_ratio = 0.75 ; // ratio of depth features over color features to sample
};

struct TrainingParametersRegression {
    unsigned int reservoir_size = 50 ; // maximum size of votes accumulated per leaf and joint
    float ms_learning_sigma = 0.05 ; // bandwidth of mean shift clustering of joint offsets per leaf (eq. 12)
    unsigned int K = 2 ;          // modes of votes to store per leaf and joint
};

using cvx::util::IBinaryStream ;
using cvx::util::OBinaryStream ;
using cvx::util::PinholeCamera ;
using cvx::util::RNG ;

struct Feature {

    enum Type { COLOR = 0, DEPTH = 1 } ;

    // compute feature response for given data sample
    float response(const Dataset &ds, sample_idx_t idx, bool training) ;

    // random sample a feature from the dataset
    void sample(const Dataset &ds, const TrainingParametersStructure &params, RNG &rng) ;

    void read(IBinaryStream &ar) {
        ar >> type_ >> ux_ >> uy_ >> vx_ >> vy_ ;
        if ( type_ == COLOR ) ar >> c1_ >> c2_ ;
    }
    void write(OBinaryStream &ar) {
        ar << type_ << ux_ << uy_ << vx_ << vy_ ;
        if ( type_ == COLOR ) ar << c1_ << c2_ ;
    }

    int16_t ux_, uy_, vx_, vy_, c1_, c2_, type_ ;
};
typedef uint64_t HistogramKey ;
typedef std::map<HistogramKey, uint64_t > Histogram ;

class RegressionModel {
public:

    RegressionModel(): sample_count_(0), n_bins_x_(5), n_bins_y_(5), n_bins_z_(5), n_background_(0) {}

    uint64_t sampleCount() const { return sample_count_ ; }

    double entropy(const Dataset &) const ;
    // add feature to model
    void aggregate(const Dataset &data, sample_idx_t idx) ;
    void aggregate(const Dataset &data, subset_t &sub, sample_idx_t bidx, sample_idx_t eidx) ;

    // compute the information gain of replacing this model with the partioned models (e.g. left, right nodes)
    double gain(const Dataset &, const RegressionModel &leftChild, const RegressionModel& rightChild) ;

    void read(IBinaryStream &ar) {
         ar >> sample_count_ >> n_background_ >> classes_ >> coordinates_ ;
    }

    void write(OBinaryStream &ar) {
         ar << sample_count_ << n_background_ << classes_ << coordinates_ ;
    }

    Histogram histogram_ ; // histogram of quantized object coordinates per object id (foreground samples)
    uint64_t n_background_ ; // number of background samples
    uint16_t n_bins_x_, n_bins_y_, n_bins_z_ ;
    uint32_t sample_count_ ;

    // leaf data

    std::map<std::string, Eigen::Vector3f > coordinates_ ;
    std::map<std::string, uint64_t> classes_ ;

    double entropy_ ;
};

#define MAX_TREES 10

class RandomForest
{
public:

    RandomForest() ;
    ~RandomForest() ;

     // trains a forest on the dataset with given parameters
    void train(const Dataset &ds, const TrainingParametersStructure &params) ;

    // train only a single tree (usefull if you want to run this on several PCs)
    void trainSingleTree(const Dataset &ds, const TrainingParametersStructure &params);

    struct Node ;

    // pass a dataset through the forest to assign each feature to a leaf in the forest
    void apply(const Dataset &ds, std::vector<std::vector< Node *> > &nodes_for_leafs) ;

    void apply_parallel(const Dataset &ds, std::vector<std::vector<Node *> > &leaf_node_indices);

    void retrofit(const Dataset &ds, const PinholeCamera &cam, const TrainingParametersRegression &params) ;

    uint32_t numTrees() const { return trees_.size() ; }

    // read/write

    void read(const std::string &data_path, const std::string &prefix = "hprf_") ;
    void write(const std::string &data_path, const std::string &prefix = "hprf_") ;

    struct Node {

        enum Type { Leaf = 0, Split = 1, Empty = 2 } ;

        Type type_ ; // node is leaf
        Feature feature_ ; // feature for this node
        RegressionModel data_ ;       // training data for this node
        double threshold_ ;// threshold
        Node *left_, *right_ ;

        Node(): type_(Empty), left_(0), right_(0) {}

        void write(OBinaryStream &bs)
        {
            bs.write((uint8_t)type_) ;
            if ( type_ == Split ) {
                feature_.write(bs) ;
                bs.write(threshold_) ;
            }
            else if ( type_ == Leaf )
                data_.write(bs) ;
        }

        void read(IBinaryStream &bs)
        {
            uint8_t t ;
            bs.read(t) ;

            type_ = (Type)t ;

            if ( type_ == Split ) {
                feature_.read(bs) ;
                bs.read(threshold_) ;
            }
            else if ( type_ == Leaf )
                data_.read(bs) ;
        }

        void makeLeaf() {
            type_ = Leaf ;
            left_ = 0 ;
            right_ = 0 ;
        }

        void makeSplit(const Feature &wl, double thresh) {
            feature_ = wl ;
            threshold_ = thresh ;
            type_ = Split ;
            left_ = new Node ;
            right_ = new Node ;
        }
    } ;

    class Tree {

    public:
        ~Tree() { delete root_ ; }

        void train(const Dataset &ds, subset_t &subset, sample_idx_t bidx, sample_idx_t eidx,
                   const TrainingParametersStructure &params, RNG &g) ;
        void apply(const Dataset &ds, std::vector<Node *> &leaf_node_indices) ;
        void apply_parallel(const Dataset &ds, std::vector<Node *> &indices);

        void write(OBinaryStream &strm) {
            write_node(root_, strm) ;
        }

        void read(IBinaryStream &strm) {
            root_ = new Node ;
            read_node(root_, strm) ;
        }


    private:

        friend class RandomForest ;

        struct TrainingContext {
            TrainingContext(const Dataset &ds, const TrainingParametersStructure &params, subset_t &sub, RNG &g):
                params_(params), ds_(ds), sub_(sub), rng_(g) {}
            const TrainingParametersStructure &params_ ;
            const Dataset &ds_ ;
            subset_t &sub_ ;
            RNG &rng_ ;
        };

        struct ApplyContext {
            ApplyContext(const Dataset &ds, subset_t &sub):
                ds_(ds), sub_(sub) {}
            const Dataset &ds_ ;
            subset_t &sub_ ;
        };


        void split(TrainingContext &ctx, Node *node, sample_idx_t bidx, sample_idx_t eidx, uint depth) ;
        void visit(ApplyContext &ctx, Node *node, sample_idx_t bidx, sample_idx_t eidx, std::vector<Node *> &leaf_node_indices) ;
        static int partition(std::vector<bool>& decisions, subset_t&  subset, sample_idx_t bidx, sample_idx_t eidx) ;

        void read_node(Node *node, IBinaryStream &strm)  ;
        void write_node(Node *node, OBinaryStream &strm) ;

        Node *root_ ;
        uint32_t depth_ ;
    };

    RNG rng_ ;

public:

    std::vector<Tree *> trees_ ;


};



#endif
