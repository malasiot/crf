#include "rf.hpp"

#include <cvx/util/misc/path.hpp>
#include <cvx/util/misc/sequence.hpp>
#include <cvx/util/misc/strings.hpp>

#include <fstream>

#include "util.hpp"

using namespace cvx::util ;
using namespace std ;
using namespace Eigen ;

RandomForest::RandomForest() {
}

void RandomForest::read(const string &data_path, const string &prefix)
{
    vector<string> tree_files = Path::glob(data_path, prefix + "*") ;

    for ( const string &tree_file: tree_files )
    {
        ifstream strm(tree_file, ios::binary) ;
        IBinaryStream ar(strm) ;

        Tree *t = new Tree() ;
        t->read(ar) ;
        trees_.push_back(t) ;
    }
}

void RandomForest::write(const string &data_path, const string &prefix)
{
    FileSequence tseq(prefix, ".bin", 2) ;

    for(uint i=0 ; i<trees_.size() ; i++)  {
        Path fpath(data_path, tseq.format(i)) ;
        ofstream strm(fpath.toString(), ios::binary) ;
        OBinaryStream ar(strm) ;

        trees_[i]->write(ar) ;
    }
}

RandomForest::~RandomForest() {
    for( int i=0 ; i<trees_.size() ; i++ )
        delete trees_[i] ;
}


void RandomForest::train(const Dataset &ds, const TrainingParametersStructure &params)
{
    for ( int i = 0; i < params.num_trees; i++ )  {
        trainSingleTree(ds, params) ;
    }
}


void RandomForest::trainSingleTree(const Dataset &ds, const TrainingParametersStructure &params)
{
    sample_idx_t n = ds.numSamples() ;

    subset_t subset ;
    rng_.sample((uint32_t)std::min((sample_idx_t)params.num_samples_per_tree, n), n, subset) ;

    Tree *t = new Tree() ;
    t->train(ds, subset, 0, subset.size(), params, rng_) ;
    trees_.push_back(t) ;
}

void RandomForest::apply(const Dataset &ds, std::vector< std::vector<Node *> > &leaf_node_indices)
{
    leaf_node_indices.resize(trees_.size()) ;


#pragma omp parallel for
    for ( int i = 0; i < trees_.size(); i++ )
    {
        Tree *t = trees_[i] ;

        std::vector<Node *> indices ;
        sample_idx_t n = ds.numSamples() ;
        indices.resize(n) ;

        t->apply(ds, indices) ;
        leaf_node_indices[i] = indices ;
    }

}

void RandomForest::apply_parallel(const Dataset &ds, std::vector< std::vector<Node *> > &leaf_node_indices)
{
    leaf_node_indices.resize(trees_.size()) ;

    for ( int i = 0; i < trees_.size(); i++ )
    {
        Tree *t = trees_[i] ;

        std::vector<Node *> indices ;
        sample_idx_t n = ds.numSamples() ;
        indices.resize(n) ;

        t->apply_parallel(ds, indices) ;
        leaf_node_indices[i] = indices ;
    }

}

void RandomForest::Tree::read_node(Node *node, IBinaryStream &strm) {
    node->read(strm) ;

    if ( node->type_ == Node::Split ) {
        node->left_ = new Node ;
        node->right_ = new Node ;
        read_node(node->left_, strm) ;
        read_node(node->right_, strm) ;
    }
}

void RandomForest::Tree::write_node(Node *node, OBinaryStream &strm) {
    node->write(strm) ;

    if ( node->type_ == Node::Split )  {
        write_node(node->left_, strm) ;
        write_node(node->right_, strm) ;
    }
}


void RandomForest::Tree::train(const Dataset &ds, subset_t &subset, sample_idx_t bidx, sample_idx_t eidx,
                               const TrainingParametersStructure &params, RNG &rng)
{
    TrainingContext ctx(ds, params, subset, rng) ;

    root_ = new Node ;
    split(ctx, root_, bidx, eidx, 0) ;
}

void RandomForest::Tree::apply(const Dataset &ds, vector<Node *> &indices)
{
    sample_idx_t n = ds.numSamples() ;

    // initialize dataset subset to include all data points
    subset_t subset ;

    subset.resize(n);

    for( sample_idx_t i=0 ; i<n ; i++ )
        subset[i] = i ;

    ApplyContext ctx(ds, subset) ;

    visit(ctx, root_, 0, n, indices) ;

}

void RandomForest::Tree::apply_parallel(const Dataset &ds, vector<Node *> &indices)
{
    sample_idx_t n = ds.numSamples() ;

    uint mthreads = omp_get_max_threads() ;
    uint chunk = ceil(n/(float)mthreads) ;

#pragma omp parallel for
    for(uint i=0 ; i<mthreads ; i++)
    {
        uint bidx = chunk*i ;
        uint eidx = std::min((uint)n, chunk*(i+1)) ;

        // initialize dataset subset to include all data points
        subset_t subset ;
        subset.resize(eidx-bidx);

        for( sample_idx_t j=0 ; j<eidx-bidx ; j++ )
            subset[j] = bidx + j ;

        ApplyContext ctx(ds, subset) ;

        visit(ctx, root_, 0, eidx-bidx, indices) ;

    }


}

void RandomForest::Tree::visit(ApplyContext &ctx, Node *nd, sample_idx_t bidx, sample_idx_t eidx, vector<Node *> &leaf_node_indices)
{
    if ( nd->type_ == Node::Leaf ) {
        for (sample_idx_t i = bidx; i < eidx; i++)
            leaf_node_indices[ctx.sub_[i]] = nd ;
    }
    else
    {
        if ( bidx != eidx )
        {
            sample_idx_t n = eidx - bidx ;

            vector<bool> decisions(n) ;

            for ( int i = bidx; i < eidx; i++ )
                decisions[i - bidx] = ( nd->feature_.response(ctx.ds_, ctx.sub_[i], false) < nd->threshold_) ;

            int split_idx = partition(decisions, ctx.sub_, bidx, eidx) ;

            visit(ctx, nd->left_, bidx, split_idx, leaf_node_indices) ;
            visit(ctx, nd->right_, split_idx, eidx, leaf_node_indices) ;
        }
    }
}

// partition the dataset subset based on provided decisions (true goes left, false goes right)

int RandomForest::Tree::partition(std::vector<bool>& decisions, subset_t&  subset, sample_idx_t bidx, sample_idx_t eidx)
{
    sample_idx_t i = bidx;
    sample_idx_t j = eidx - 1;

    while (i != j)
    {
        if ( !decisions[i - bidx] )
        {
            bool d = decisions[i - bidx];
            sample_idx_t idx = subset[i];

            decisions[i - bidx] = decisions[j - bidx];
            subset[i] = subset[j];

            decisions[j - bidx] = d;
            subset[j] = idx;

            j--;
        }
        else i++ ;
    }

    return decisions[i-bidx] ? i + 1 : i ;
}

void RandomForest::Tree::split(TrainingContext &ctx, Node *cnode, sample_idx_t bidx, sample_idx_t eidx, unsigned int depth)
{
    if ( depth < 10 )
        cout << "depth: " << depth << ' ' << bidx << ' ' << eidx << endl ;

    RegressionModel pm ;
    Feature bwl ;

    pm.aggregate(ctx.ds_, ctx.sub_, bidx, eidx) ;
    pm.entropy_ = pm.entropy(ctx.ds_) ;

    sample_idx_t n = eidx - bidx ;

    if ( depth == ctx.params_.max_depth || eidx - bidx < ctx.params_.min_pts_per_leaf ) // reached recursion depth
    {
        cnode->makeLeaf( ) ;
        return ;
    }

    vector<bool> bdecisions ;
    vector<float> responses ;

    double maxGain = -DBL_MAX, bestThresh ;
    bdecisions.resize(n) ;

    responses.resize(n) ;

    // find the best partioning of the data subset

    //uint nrs = ( depth < 7 ) ? 10 : ctx.params_.num_random_samples ;

    uint nrs = ctx.params_.num_random_samples ;

    for(int t=0 ; t<nrs ; t++ )
    {
        // randomize the feature

        Feature wl ;
        wl.sample(ctx.ds_, ctx.params_, ctx.rng_) ;

        // compute feature responses over the dataset

        float min_response = FLT_MAX, max_response = -FLT_MAX;

#pragma omp parallel for
        for( sample_idx_t i=bidx ; i<eidx ; i++ )
        {
            sample_idx_t data_idx = ctx.sub_[i] ;

            double res = wl.response(ctx.ds_, data_idx, true) ;

            responses[i-bidx] = res ;
        }

        for( sample_idx_t i=bidx ; i<eidx ; i++ )
        {
            float res = responses[i-bidx] ;

            min_response = std::min(min_response, res) ;
            max_response = std::max(max_response, res) ;

        }
        //       cout << min_response << ' ' << max_response << endl ;
        // choose candidate thresholds and optimize

        vector<float> thresholds ;

        if ( fabs(min_response-max_response ) < 1.0e-10 )
        {
            thresholds.push_back(min_response - 1.0e-3 ) ;
            thresholds.push_back(max_response + 1.0e-3 ) ;
        }
        else {

            assert(min_response < max_response) ;

            for ( unsigned int k=0 ; k<ctx.params_.num_thresholds_per_sample ; k++ )
            {
                float thresh = ctx.rng_.uniform(min_response, max_response) ;
                thresholds.push_back(thresh) ;
            }

        }

#pragma omp parallel for
        for ( unsigned int k=0 ; k<thresholds.size() ; k++ )
        {
            RegressionModel lm, rm ;
            std::vector<bool> decisions(n) ;

            float thresh = thresholds[k] ;

            for( unsigned int i=bidx ; i<eidx ; i++ )
            {
                sample_idx_t data_idx = ctx.sub_[i] ;

                if ( responses[i-bidx] < thresh ) {
                    lm.aggregate(ctx.ds_, data_idx) ;
                    decisions[i - bidx] = true ;
                }
                else {
                    rm.aggregate(ctx.ds_, data_idx) ;
                    decisions[i - bidx] = false ;
                }
            }

            double gain = pm.gain(ctx.ds_, lm, rm) ;

#pragma omp critical
            if ( gain >= maxGain )
            {
                maxGain = gain;
                bwl = wl ;
                std::copy(decisions.begin(), decisions.end(), bdecisions.begin()) ;
                bestThresh = thresh ;
            }

        }

    }

    //    if ( depth < 12 ) cout << maxGain << endl ;


    if ( maxGain < ctx.params_.gain_threshold  ) // leaf node
        cnode->makeLeaf( ) ;
    else { // split node

        sample_idx_t split_idx = partition(bdecisions, ctx.sub_, bidx, eidx) ;

        cnode->makeSplit( bwl, bestThresh ) ;

        // split data subset and recurse to left and right node

        split(ctx, cnode->left_, bidx, split_idx, depth + 1) ;
        split(ctx, cnode->right_, split_idx, eidx, depth + 1) ;
    }
}
////////////////////////////////////////////////////////////////////////////////

typedef map<string, vector<Vector3f> > CoordMap ;

void RandomForest::retrofit(const Dataset &ds, const PinholeCamera &cam, const TrainingParametersRegression &params)
{
    // find leafs reached by the dataset
    vector< vector<Node *> > leafs ;

    apply_parallel(ds, leafs) ;

    map<Node *, CoordMap > node_map ;

    for(uint t=0 ;t<numTrees() ; t++)
    {
        for( uint i=0 ; i<leafs[t].size() ; i++ )
        {
            Node *n = leafs[t][i] ;

            string label = ds.labels_[i] ;
            const Vector3f &coords = ds.coordinates_[i] ;

            n->data_.sample_count_ ++ ;

            if ( label.empty() )
                n->data_.n_background_ ++ ;
            else {
                map<string, uint64_t>::iterator it = n->data_.classes_.find(label) ;
                if ( it == n->data_.classes_.end() ) n->data_.classes_.insert(std::make_pair(label, (uint64_t)1)) ;
                else it->second ++ ;

                map<Node *, CoordMap>::iterator cit = node_map.find(n) ;
                if ( cit == node_map.end() ) node_map[n][label].push_back(coords) ;
                else {
                    (cit->second)[label].push_back(coords) ;
                }
            }
        }
    }


    // perform mean-shift one coordinates acuumulated at each leaf

    map<Node *, CoordMap>::const_iterator it = node_map.begin() ;

    for( ; it != node_map.end() ; ++it ) {
        Node *n = it->first ;
        const CoordMap &coord_map = it->second ;

        CoordMap::const_iterator cit = coord_map.begin() ;

        for( ; cit != coord_map.end() ; ++cit ) {
            string label = cit->first ;
            const vector<Vector3f> &coords = cit->second ;

            float weight ;
            Vector3f mode ;
            mean_shift(rng_, coords, mode, weight, params.ms_learning_sigma, 30, 1.0, 10) ;

            n->data_.coordinates_[label] = mode ;

        }
    }

}

////////////////////////////////////////////////////////////////////////////////

const ushort DEPTH_MAX = 20000 ;

float Feature::response(const Dataset &ds, sample_idx_t idx, bool training)
{
    if ( type_ == DEPTH ) {
        cv::Mat_<ushort> im = ds.getDepthImageD(idx) ;
        const cv::Point &p = ds.getPoint(idx) ;

        ushort val = im[p.y][p.x] ;
        float val1, val2 ;

        int ux = p.x + ux_ * 1000.0 / val ;
        int uy = p.y + uy_ * 1000.0 / val ;
        int vx = p.x + vx_ * 1000.0 / val ;
        int vy = p.y + vy_ * 1000.0 / val ;

        if ( ux < 0 || uy < 0 || ux > im.cols-1 || uy > im.rows-1 ) val1 = DEPTH_MAX ;
        else {
            val1 = im[uy][ux] ;
            if ( val1 == 0 ) val1 = DEPTH_MAX ;
        }

        if ( vx < 0 || vy < 0 || vx > im.cols-1 || vy > im.rows-1 ) val2 = DEPTH_MAX ;
        else {
            val2 = im[vy][vx] ;
            if ( val2 == 0 ) val2 = DEPTH_MAX ;
        }

        return ( val1 - val2 )/1000.0 ;
    }
    else {
        cv::Mat_<cv::Vec3b> rgb = ds.getColorImage(idx) ;
        cv::Mat_<ushort> depth = ds.getDepthImageC(idx) ;
        const cv::Point &p = ds.getPoint(idx) ;

        ushort val = depth[p.y][p.x] ;
        float val1, val2 ;

        int ux = p.x + ux_ * 1000.0 / val ;
        int uy = p.y + uy_ * 1000.0 / val ;
        int vx = p.x + vx_ * 1000.0 / val ;
        int vy = p.y + vy_ * 1000.0 / val ;

        if ( ux < 0 || uy < 0 || ux > rgb.cols-1 || uy > rgb.rows-1 ) val1 = 0 ;
        else  {
            ushort dv = depth[uy][ux] ;
            if ( dv != 0 ) val1 = rgb[uy][ux][c1_] ;
            else val1 = 0 ;
        }

        if ( vx < 0 || vy < 0 || vx > rgb.cols-1 || vy > rgb.rows-1 ) val2 = 0 ;
        else {
            ushort dv = depth[vy][vx] ;
            if ( dv != 0 ) val2 = rgb[vy][vx][c2_] ;
            else val2 = 0 ;
        }

        return ( val1 - val2 )/255.0 ;
    }
}

void Feature::sample(const Dataset &ds, const TrainingParametersStructure &params, RNG &rng)
{

    type_ = ( rng.uniform(0.0, 1.0) < params.depth_to_color_ratio ) ? DEPTH : COLOR ;

    if ( type_ == DEPTH ) {

        while (1) {
            ux_ = rng.uniform<int>(-params.max_probe_offset_depth, params.max_probe_offset_depth) ;
            uy_ = rng.uniform<int>(-params.max_probe_offset_depth, params.max_probe_offset_depth) ;
            if ( ux_*ux_ + uy_*uy_ < params.max_probe_offset_depth * params.max_probe_offset_depth ) break ;
        }

        while (1) {
            vx_ = rng.uniform<int>(-params.max_probe_offset_depth, params.max_probe_offset_depth) ;
            vy_ = rng.uniform<int>(-params.max_probe_offset_depth, params.max_probe_offset_depth) ;
            if ( vx_*vx_ + vy_*vy_ < params.max_probe_offset_depth * params.max_probe_offset_depth ) break ;
        }
    }
    else {
        c1_ = rng.uniform<int>(0, 2) ;
        c2_ = rng.uniform<int>(0, 2) ;

        while (1) {
            ux_ = rng.uniform<int>(-params.max_probe_offset_rgb, params.max_probe_offset_rgb) ;
            uy_ = rng.uniform<int>(-params.max_probe_offset_rgb, params.max_probe_offset_rgb) ;
            if ( ux_*ux_ + uy_*uy_ < params.max_probe_offset_rgb * params.max_probe_offset_rgb ) break ;
        }

        while (1) {
            vx_ = rng.uniform<int>(-params.max_probe_offset_rgb, params.max_probe_offset_rgb) ;
            vy_ = rng.uniform<int>(-params.max_probe_offset_rgb, params.max_probe_offset_rgb) ;
            if ( vx_*vx_ + vy_*vy_ < params.max_probe_offset_rgb * params.max_probe_offset_rgb ) break ;
        }
    }
}

double RegressionModel::entropy(const Dataset &data) const
{
    float sum = 0.0 ;

    Histogram::const_iterator it = histogram_.begin() ;
    for( ; it != histogram_.end() ; ++it )
    {
        float p = it->second / (float)sample_count_;

        if (p > 0)
            sum -= p * log(p);
    }

    if ( n_background_ > 0 ) {
        float p = n_background_/(float)sample_count_ ;
        sum -= p * log(p) ;
    }

    return sum ;
}
// update histogram

void RegressionModel::aggregate(const Dataset &data, sample_idx_t idx)
{
    sample_count_ ++ ;

    string label = data.labels_[idx] ;

    if ( label.empty() ) {
        n_background_ ++ ;
        return ;
    }

    uint16_t label_id = std::distance(data.label_map_.begin(), std::find(data.label_map_.begin(), data.label_map_.end(), label)) ;

    uint32_t image_idx = data.image_idx_[idx] ;
    Vector3f p = data.coordinates_[idx] ;
    Vector3f extents = data.boxes_[image_idx] ;

    float bin_x = p.x() / extents.x() + 0.5 ;
    float bin_y = p.y() / extents.y() + 0.5 ;
    float bin_z = p.z() / extents.z() + 0.5 ;

    uint q_bin_x, q_bin_y, q_bin_z ;


    if ( bin_x < 0 || bin_x >= 1.0 ) return ;
    else q_bin_x = n_bins_x_ * bin_x ;

    if ( bin_y < 0 || bin_y >= 1.0 ) {
        cout << p.adjoint() << endl ;
        return ;
    }
    else q_bin_y = n_bins_y_ * bin_y ;

    if ( bin_z < 0 || bin_z >= 1.0 ) return ;
    else q_bin_z = n_bins_z_ * bin_z ;

    HistogramKey key =  ((label_id*n_bins_z_ + q_bin_z)*n_bins_y_ + q_bin_y)*n_bins_x_ + q_bin_x;

    Histogram::iterator it = histogram_.find(key) ;

    if ( it == histogram_.end() ) histogram_.insert(make_pair(key, (uint64_t)1)) ;
    else it->second ++ ;

}

#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
void RegressionModel::aggregate(const Dataset &data, subset_t &sub, sample_idx_t bidx, sample_idx_t eidx)
{
    for(sample_idx_t idx=bidx; idx<eidx; idx++) {
        uint32_t pidx = sub[idx] ;
        aggregate(data, pidx) ;
    }
}


double RegressionModel::gain(const Dataset &ds, const RegressionModel &leftChild, const RegressionModel &rightChild)
{
    //   double entropyBefore = entropy(ds);
    double entropyBefore = entropy_;

    const uint64_t n1 = leftChild.sampleCount() ;
    const uint64_t n2 = rightChild.sampleCount() ;

    uint64_t totalSamples = n1 + n2;

    if ( totalSamples <= 1 ) return 0 ;

    double entropyAfter = (n1 * leftChild.entropy(ds) + n2 * rightChild.entropy(ds))/totalSamples ;

    return entropyBefore - entropyAfter;
}




