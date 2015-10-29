#ifndef MMSB_MMSB_MODEL_HPP_
#define MMSB_MMSB_MODEL_HPP_

#include "vertex.hpp"
#include "context.hpp"
#include "alias_table.hpp"

#include <ctime>

namespace mmsb {

class MMSBModel {
 public:
  MMSBModel(const ModelParameter& param,
      const WIndex client_id, const WIndex thread_id);
  MMSBModel(const string& param_file);

  void Solve(const char* resume_file = NULL);
  inline void Solve(const string& resume_file) {
    Solve(resume_file.c_str());
  }

  void ReadData(); //TODO

  void Snapshot();
  void SnapshotVis(); // temp: for visualization/debug
  void ToProto(ModelParameter* param);
  void Restore(const char* snapshot_file = NULL);

  inline uint32 K() const { return K_; }
  inline const vector<pair<float, float> >& lambda() const { return lambda_; }

 private:
  void SampleMinibatch(unordered_set<VIndex>& vertex_batch,
    set<pair<VIndex, VIndex> >& link_batch,
    const Count batch_size);
  void CollectLinks(set<pair<VIndex, VIndex> >& link_batch,
    const unordered_set<VIndex>& vertex_batch);
  void GibbsSample();
  void MHSample();
  void MHSampleLinkRemote(const VIndex i, const VIndex j, const CIndex z_prev);
  void MHSampleLinkFeedback(const VIndex j, const VIndex nbr_j,
      const CIndex z_prev, const CIndex betap, const CIndex nbrp);
  void MHSampleAcptRjct(const VIndex i, const VIndex j,
      const CIndex jp, const float* ratio_jterms);
  CIndex MHSampleLink(const Vertex* v_i, const Vertex* v_j, const CIndex z_prev);
  CIndex MHSampleWithVertexProposal(const Vertex* v_a, const Vertex* v_b,
      const CIndex z_prev, const CIndex s);
  void VIComputeGrads();
  void VIUpdate();
  void Test();

  void Init(const ModelParameter& param);
  void InitMsgPassing();
  void InitModelState();

  void EstimateBeta();
  void ComputeExpELogBeta(); //TODO
  float ComputeLinkLikelihood(const VIndex i, const VIndex j,
      const bool positive);
  inline float MembershipProb(const Count cnt, const Count degree);

  // worker id of the neighbor vertices or the vertices on this worker
  inline WIndex WorkerId(const VIndex i);
  inline uint32 GetOMsgRowId(const VIndex nbr_w);
  inline uint32 GetIMsgRowId(const VIndex nbr_w);
  inline void AdvanceIMsgRowId(const VIndex nbr_w);
  inline void ResetMsgRow(const uint32 rid);

  // set z of neighbor vertices via msg passing
  void SendZMsg(VIndex nbr_i, Vindex i, CIndex z);
  void SetZfromMsg();
  float ComputeMHRatioTerm(Vertex* v, CIndex s, CIndex t,
      CIndex z_prev, int mh_step);

 private:
  ModelParameter param_;

  unordered_map<VIndex, Vertex*> vertices_; // vertices stored on this worker
  unordered_map<VIndex, WIndex> neighbor_worker_; // neighbor vertex id to worker id
 
  uint32 K_; // #communities
  float epsilon_;
  vector<float> beta_; // community strength  
  /// <lambda_0, lambda_1>
  //vector<pair<float, float> > lambda_; // prior for beta_
  vector<float> lambda_0_; // prior for beta_
  vector<float> lambda_1_; // prior for beta_
  /// gradients
  vector<pair<float, float> > lambda_grads_; 
  /// intermediate factors for computing gradients
  vector<float> bphi_sum_;
  vector<float> bphi_square_sum_;
   
  pair<float, float> eta_; // prior of community strength
  float alpha_; // prior of community membership

  /* -------------------- Training --------------------- */
  int iter_;
  float lr_; // learning rate
  Count train_batch_size_;
  unordered_set<VIndex> train_batch_vertices_;
  set<pair<VIndex, VIndex> > train_batch_links_;
  vector<CIndex> train_batch_k_cnts_; // statistics of the samples
  unordered_map<VIndex, pair<CIndex, CIndex> > beta_vi_prpsls_;

  /// test data (postive and negative links)
  vector<pair<VIndex, VIndex> > test_links_;
  vector<int> test_link_values_;
  unordered_map<VIndex, WIndex> test_neighbor_worker_; // neighbor vertex id to worker id
  //Count test_batch_link_size_;
  //vector<pair<VIndex, VIndex> > test_batch_links_;
  
  // For sampling, size = K_
  vector<float> comm_probs_;
  vector<float> exp_Elog_beta_;
  AliasTable beta_alias_table_;

  clock_t start_time_;

  // temp: for debug
  Count beta_proposal_accept_times_;
  Count beta_proposal_times_;
  Count vertex_proposal_accept_times_;
  Count vertex_proposal_times_;

  /* ------------------ Msg passing ------------------- */
  
  WIndex client_id_;
  WIndex thread_id_;
  WIndex worker_id_;
  uint32 num_workers_;

  petuum::Table<float> msg_table_;
  petuum::Table<float> param_table_;

  // outgoing msg to worker w is between 
  // row omsg_st_rid_[w] and row omsg_ed_rid_[w]-1 of the msg-table
  vector<uint32> omsg_st_rid_;
  vector<uint32> omsg_ed_rid_;
  vector<uint32> omsg_cur_rid_;
  // incoming msg from worker w is between 
  // row omsg_st_rid_[w] and row omsg_ed_rid_[w]-1 of the msg-table
  vector<uint32> imsg_st_rid_;
  vector<uint32> imsg_ed_rid_;
  vector<uint32> imsg_cur_rid_;
};
 
inline float MMSBModel::MembershipProb(
    const Count cnt, const Count degree) {
  return (cnt + alpha_) / (degree + K_ * alpha_);
}

inline uint32 GetOMsgRowId(const WIndex nbr_w) {
  uint32& row_id = omsg_cur_rid_[nbr_w];
  uint32 ret = row_id;  
  row_id++;
  row_id = row_id < omsg_ed_rid_[nbr_w] ? row_id : omsg_st_rid_[nbr_w];
  return ret;
}
inline void AdvanceIMsgRowId(const WIndex nbr_w) {
  uint32& row_id = imsg_cur_rid_[nbr_w];
  row_id++;
  row_id = row_id < imsg_ed_rid_[nbr_w] ? row_id : imsg_st_rid_[nbr_w];
}

inline WIndex MMSBModel::WorkerId(const VIndex i) {
  auto iter = neighbor_worker_.find(i);
  if (iter != neighbor_worker_.end) {
    return iter->second;
  }
  return worker_id_;
}

inline void MMSBModel::ResetMsgRow(const uint32 rid) {
  petuum::DenseUpdateBatch<float> update_batch(0, 1);
  update_batch[kColIdxMsgType] = kInactive;
  msg_table_.DenseBatchInc(rid, update_batch);
}

} // namespace mmsb

#endif
