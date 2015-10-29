#ifndef MMSB_VERTEX_HPP_
#define MMSB_VERTEX_HPP_

#include "common.hpp"

namespace mmsb {

class Vertex {
 public:
  Vertex() { }

  inline void SetZ(const VIndex neighbor_id, const CIndex z) {
    CHECK(neighbor_z_.find(neighbor_id) != neighbor_z_.end());
    neighbor_z_[neighbor_id] = z;

    //TODO(zhiting): may increase memory consumption
    z_cnts_[z] += 1;
  }
  inline void RemoveZ(const CIndex z) {
    CHECK(z_cnts_.find(z) != z_cnts_.end()) << z << "\t" << z_cnts_.size();
    z_cnts_[z] -= 1;
  }

  inline const unordered_map<VIndex, CIndex>& neighbor_z() const {
    return neighbor_z_;
  }
  inline CIndex z_by_vertex_id(const VIndex vid) const {
    unordered_map<CIndex, CIndex>::const_iterator it = neighbor_z_.find(vid);
    CHECK(it != neighbor_z_.end());
    return it->second;
  }
  inline CIndex z_by_neighbor_idx(const uint32 neighbor_idx) const {
    CHECK_LT(neighbor_idx, neighbors_.size());
    VIndex vid = neighbors_[neighbor_idx];
    return z_by_vertex_id(vid);
  }
  inline pair<CIndex, CIndex>& nbr_betav_prpsls(const VIndex nbr) {
#ifdef DEBUG
     CHECK(nbr_betav_prpsls_.find(nbr) != nbr_betav_prpsls_.end());
#endif
     return nbr_betav_prpsls_[nbr];
  }
  inline const unordered_map<CIndex, Count>& z_cnts() const {
    return z_cnts_;
  }
  inline float z_cnt(const CIndex z) const {
    unordered_map<CIndex, Count>::const_iterator it = z_cnts_.find(z);
    return (it == z_cnts_.end() ? 0 : it->second);
  }
  inline Count degree() const {
    return degree_;
  }

  inline bool IsNeighbor(const VIndex vid) {
    return neighbor_z_.find(vid) != neighbor_z_.end();
  }
  inline void AddLink(const VIndex vid, const bool is_remote) {
    neighbor_z_[vid] = 0;
    degree_ = neighbor_z_.size();
    if (degree_ > neighbors_.size()) { // to avoid duplicate in neighbors_
      neighbors_.push_back(vid);
    }
#ifdef DEBUG
    CHECK_EQ(neighbors_.size(), degree_);
#endif
    if (is_remote) {
      nbr_betav_prpsls_[vid] = pair<CIndex, CIndex>();
    }
  }

  void ToProto(VertexParameter* param);
  void FromProto(const VertexParameter& param);

 private:
  //VIndex index_;
  
  Count degree_;
  vector<VIndex> neighbors_; // used in MH Sampler
  unordered_map<VIndex, CIndex> neighbor_z_; // neighbor vertex id => z
  unordered_map<CIndex, Count> z_cnts_; // summary of z_, community k => #{z = k}
  //float coeff_; // =(N-1)/(degree-1); =0 if degree == 0
  
  /// for distributed training
  // remote neighbor vertex id => (beta-proposal, i-proposal)
  unordered_map<VIndex, pair<CIndex, CIndex> > nbr_betav_prpsls_;
};
 
} // namespace mmsb

#endif
