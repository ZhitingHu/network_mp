
#include "io.hpp"
#include "util.hpp"
#include "mmsb_model.hpp"

namespace mmsb {

MMSBModel::MMSBModel(const ModelParameter& param,
    const WIndex client_id, const WIndex thread_id) : 
    client_id_(client_id), thread_id_(thread_id) {
  Init(param);
}
MMSBModel::MMSBModel(const string& param_file,
    const WIndex client_id, const WIndex thread_id) :
    client_id_(client_id), thread_id_(thread_id) {
  ModelParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

void MMSBModel::Init(const ModelParameter& param) {
  param_ = param;
  /// 
  K_ = param_.num_comms();
  epsilon_ = param_.epsilon();
  eta_.first = param_.eta_0();
  eta_.second = param_.eta_1();
  //alpha_ = param_.alpha();
  alpha_ = 1.0 / K_;
  beta_.resize(K_);
  lr_ = param_.solver_param().base_lr();
  lambda_0_.resize(K_);
  lambda_1_.resize(K_);
  lambda_grads_.resize(K_);
  bphi_sum_.resize(K_);
  bphi_square_sum_.resize(K_);
  /// minibatch 
  train_batch_size_ = param_.solver_param().train_batch_size();
  train_batch_k_cnts_.resize(K_);
  /// sampling
  exp_Elog_beta_.resize(K_);
  comm_probs_.resize(K_);

  /// temp
  beta_proposal_accept_times_ = 0;
  beta_proposal_times_ = 0;
  vertex_proposal_accept_times_ = 0;
  vertex_proposal_times_ = 0;

  InitMsgPassing();
}

void MMSBModel::InitMsgPassing() {
  //
  msg_table_ = petuum::PSTableGroup::GetTableOrDie<float>(kMsgTableID);
  param_table_ = petuum::PSTableGroup::GetTableOrDie<float>(kParamTableID);
  //
  uint32 num_clients = mmsb::Context::get_int32("num_clients");
  uint32 num_threads = mmsb::Context::get_int32("num_app_threads");
  num_workers_ = num_clients * num_threads; 
  worker_id_ = client_id_ * num_threads + thread_id_;
  omsg_st_rid_.resize(num_workers);
  omsg_ed_rid_.resize(num_workers);
  imsg_st_rid_.resize(num_workers);
  imsg_ed_rid_.resize(num_workers);
  uint32 num_imsg_rows_per_worker = kMsgTabMaxNumRows / num_workers;
  uint32 num_omsg_rows_per_worker_pair = num_imsg_rows_per_worker / num_workers;
  for (WIndex w = 0; w < num_workers; ++w) {
    if (w == worker_id_) continue;
    // outgoing msg
    omsg_st_rid_[w] = num_imsg_rows_per_worker * w
        + num_omsg_rows_per_worker_pair * worker_id_;
    omsg_ed_rid_[w] = omsg_st_rid_[w] + num_omsg_rows_per_worker_pair;
    omsg_cur_rid_[w] = omsg_st_rid_[w];
    // incoming msg
    imsg_st_rid_[w] = num_imsg_rows_per_worker * worker_id_
        + num_omsg_rows_per_worker_pair * w;
    imsg_ed_rid_[w] = imsg_st_rid_[w] + num_omsg_rows_per_worker_pair;
    imsg_cur_rid_[w] = imsg_st_rid_[w];
  } 
}

/// Input format: from_v_id \t to_v_id
void MMSBModel::ReadData() { //TODO
  /// training data
  const string& train_data_path = Context::get_string("train_data");
  LOG(INFO) << "Read train data from " << train_data_path;
  fstream train_data_file(train_data_path.c_str(), ios::in);
  CHECK(train_data_file.is_open()) << "Fail to open " << train_data_path; 
  // header: #v #e
  Count num_vertices = 0;
  train_data_file >> num_vertices;
  LOG(INFO) << "#Vertices\t" << num_vertices;
  CHECK_LE(train_batch_size_, num_vertices);
  vertices_.resize(num_vertices);
  for (VIndex i = 0; i < num_vertices; ++i) {
    vertices_[i] = new Vertex();
  }
  // links
  VIndex i, j;
  while (train_data_file >> i >> j) {
    if (i != j) { // avoid self-loop
      vertices_[i]->AddLink(j);
      vertices_[j]->AddLink(i);
    }
  }
  train_data_file.close();

  /// test data
  const string& test_data_path = Context::get_string("test_data");
  LOG(INFO) << "Read test data from " << test_data_path;
  fstream test_data_file(test_data_path.c_str(), ios::in);
  CHECK(test_data_file.is_open()) << "Fail to open " << test_data_path; 
  int value = 0;
  WIndex i_worker, j_worker;
  while (test_data_file >> i >> j >> value >> i_worker >> j_worker) {
#ifdef DEBUG
    CHECK(vertices_.find(i) != vertices_.end());
    CHECK_EQ(i_worker, worker_id_);
#endif
    test_links_.push_back(make_pair(i, j));
    test_link_values_.push_back(value);
    if (j_worker != worker_id_) {
      test_neighbor_worker_[j] = j_worker;
    }
  }
  LOG(INFO) << "#Test links (pos and neg links)\t"
      << test_links_.size();
  test_data_file.close();
}

void MMSBModel::InitModelState() {
  /// community assignment
  for (auto& id_v : vertices_) {
    const VIndex i = id_v.first;
    Vertex* v = id_v.second;
    const unordered_map<VIndex, CIndex>& neighbor_z = v->neighbor_z();
    for (const auto& nz : neighbor_z) {
      VIndex j = nz.first;
      if (i >= j) continue;

      CIndex z = Context::randUInt64() % K_;
      v->SetZ(j, z);
      auto vj = vertices_.find(j);
      if (vj != vertices_.end()) { // vj is on the local worker
        vj->second->SetZ(i, z);
      } else { // vj is on another worker, send msg
        SendZMsg(j, i, z);
      }
    }
  }

  petuum::PSTableGroup::GlobalBarrier();
  SetZfromMsg();

  /// global param
  fill(lambda_0_.begin(), lambda_0_.end(), eta_.first);  
  fill(lambda_1_.begin(), lambda_1_.end(), eta_.second);  

  LOG(INFO) << "Init model state done.";
}

void MMSBModel::SendZMsg(VIndex nbr_i, Vindex i, CIndex z) {
#ifdef DEBUG
  CHECK(neighbor_worker_.find(nbr_i) != neighbor_worker_.end());
#endif
  // row id
  WIndex nbr_w = neighbor_worker_[nbr_i];
  uint32 row_id = GetOMsgRowId(nbr_w);
  // msg content
  petuum::DenseUpdateBatch<float> update_batch(
      0, kNumMsgPrfxCols + 1);
  update_batch[kColIdxMsgType] = kSetZ;
  update_batch[kColIdxMsgVId] = nbr_i;
  update_batch[kColIdxMsgNbrId] = i;
  update_batch[kColIdxMsgSt] = z;
  // send the msg 
  msg_table_.DenseBatchInc(row_id, update_batch);
}
 
void MMSBModel::SetZfromMsg() {
  /// recv all msgs to this worker
  for (WIndex w = 0; w < num_workers_; ++w) {
    if (w == worker_id_) continue;
    uint32& rid = imsg_cur_rid_[w] = imsg_st_rid_[w];
    while(true) {
      vector<float> row_cache(kNumMsgPrfxCols + 1); //TODO: set size to the maximum size ?
      petuum::RowAccessor row_acc;
      const auto& r 
          = msg_table_.Get<petuum::DenseRow<float> >(rid, &row_acc);
      r.CopyToVector(&row_cache);
      uint32 msg_type = row_cache[kColIdxMsgType];
      if (msg_type == kSetZ) {
        VIndex j = row_cache[kColIdxMsgVId];
        VIndex nbr_j = row_cache[kColIdxMsgNbrId];
        CIndex z = row_cache[kColIdxMsgSt];
#ifdef DEBUG
        CHECK(vertices_.find(j) != vertices_.end());
        CHECK(neighbor_worker_.find(nbr_j) != neighbor_worker_.end());
#endif
        vertices_[j]->SetZ(nbr_j, z);
        
        ResetMsgRow(rid);    // reset the row 
        AdvanceIMsgRowId(w); // next msg
      } else if (msg_type == kInactive) { // empty msg, stop
        break;
      } else {
        LOG(FATAL) << "Illegial msg type " << msg_type;
      }
    } // end of rows
  }   // end of workers
}

/* -------------------- Train --------------------- */

void MMSBModel::SampleMinibatch(unordered_set<VIndex>& vertex_batch,
    set<pair<VIndex, VIndex> >& link_batch, const Count batch_size) {
  vertex_batch.clear();
  for (int i = 0; i < batch_size; ++i) {
    while (true) {
      // randomly sample a vertex from the map<.,.> vertices_
      auto it = vertices_.begin();
      uint32 nstep = Context::randUInt64() % vertices_.size();
      std::advance(it, nstep);
      VIndex v = it->first;
      if (vertex_batch.find(v) == vertex_batch.end()) { // to avoid duplicate
        vertex_batch.insert(v);
        break;
      }
    }
  }
  
  CollectLinks(link_batch, vertex_batch);

  for (const auto& link : link_batch) {
    vertex_batch.insert(link.first);
    vertex_batch.insert(link.second);
  }

  //LOG(INFO)
  //    << "iter " << iter_ << ", "
  //    << vertex_batch.size() << " vertexes, " 
  //    << link_batch.size() << " links";
}

void MMSBModel::CollectLinks(set<pair<VIndex, VIndex> >& link_batch,
    const unordered_set<VIndex>& vertex_batch) {
  link_batch.clear();
  for (const auto i : vertex_batch) {
#ifdef DEBUG
    CHECK(vertices_.find(i) != vertices_.end());
#endif
    const unordered_map<VIndex, CIndex>& neighbor_z
        = vertices_[i]->neighbor_z();
    for (const auto& nz : neighbor_z) {
      pair<VIndex, VIndex> link
          = make_pair(min(i, nz.first), max(i, nz.first));
      link_batch.insert(link);
    } // end of neighbors of i
  }   // end of vertexes
}

#if 0
void MMSBModel::GibbsSample() {
  ComputeExpELogBeta();
  fill(train_batch_k_cnts_.begin(), train_batch_k_cnts_.end(), 0);
  for (const auto& link : train_batch_links_) {
    VIndex i = link.first;
    VIndex j = link.second;
    CHECK_LT(i, vertices_.size());
    CHECK_LT(j, vertices_.size());
    Vertex* v_i = vertices_[i];
    Vertex* v_j = vertices_[j];

    CHECK_EQ(v_i->z_by_vertex_id(j), v_j->z_by_vertex_id(i));
    CIndex z_prev = v_i->z_by_vertex_id(j);
    v_i->RemoveZ(z_prev);
    v_j->RemoveZ(z_prev);
   
    // Compute distribution 
    float prob_sum = 0;
    for (int k = 0; k < K_; ++k) {
      int nik = v_i->z_cnt(k);
      int njk = v_j->z_cnt(k);
      float factor_i = alpha_ + (nik == 0 ? 0 :
          nik * (vertices_.size() - 2.0) / (v_i->degree() - 1)); // to avoid divide-by-0
      float factor_j = alpha_ + (njk == 0 ? 0 :
          njk * (vertices_.size() - 2.0) / (v_j->degree() - 1)); // to avoid divide-by-0
      comm_probs_[k] = factor_i * factor_j * exp_Elog_beta_[k];

      CHECK(!isnan(comm_probs_[k]))
          << factor_i << "\t" << factor_j << "\t" << exp_Elog_beta_[k] << "\t" 
          << lambda_[k].second << "\t" << lambda_[k].first << "\t" 
          << mmsb::digamma(lambda_[k].first) << "\t" 
          << mmsb::digamma(lambda_[k].first + lambda_[k].second) << "\t" << k;

      prob_sum += comm_probs_[k];
    }
    // Sample
    const float rand_num = Context::rand() * prob_sum;
    float part_prob_sum = 0;
    CIndex z_new = K_;
    for (int k = 0; k < K_; ++k) {
      part_prob_sum += comm_probs_[k];
      if (part_prob_sum >= rand_num) {
        z_new = k;
        break;
      }
    }
    CHECK_LT(z_new, K_) << rand_num << "\t" << prob_sum << "\t" << part_prob_sum;

    // Update
    v_i->SetZ(j, z_new);
    v_j->SetZ(i, z_new);
    train_batch_k_cnts_[z_new]++;
  } // end of batch
}
#endif

void MMSBModel::MHSample() {
  ComputeExpELogBeta();
  // Build alias table for exp{ E[ log(beta) ] }
  beta_alias_table_.BuildAliasTable(exp_Elog_beta_);

  fill(train_batch_k_cnts_.begin(), train_batch_k_cnts_.end(), 0);
  for (const auto& link : train_batch_links_) {
    VIndex i = link.first;
    VIndex j = link.second;
#ifdef DEBUG
    // i must be on the local worker
    CHECK(vertices_.find(i) != vertices_.end());
    CHECK(vertices_.find(j) != vertices_.end() ||
        neighbor_worker_.find(j) != neighbor_worker_.end());
#endif
    CIndex z_prev = v_i->z_by_vertex_id(j);
    auto it_vj = vertices_.find(j);
    if (it_vj != vertices_.end()) { // both vertices are on the local worker
      CIndex z_new = MHSampleLink(v_i, v_j, z_prev);
      if (z_new != z_prev) {
        v_i->RemoveZ(z_prev);
        v_j->RemoveZ(z_prev);
        v_i->SetZ(j, z_new);
        v_j->SetZ(i, z_new);
      }
      train_batch_k_cnts_[z_new]++;
    } else {    // vertex j is on another worker
      MHSampleLinkRemote(i, j, z_prev);
    }
  }

  petuum::PSTableGroup::GlobalBarrier();

  /// process all proposal msgs
  for (WIndex w = 0; w < num_workers_; ++w) {
    if (w == worker_id_) continue;
    uint32& rid = imsg_cur_rid_[w]
    while(true) {
      vector<float> row_cache(kNumMsgPrfxCols + 2); //TODO: set size to the maximum size ?
      petuum::RowAccessor row_acc;
      const auto& r 
          = msg_table_.Get<petuum::DenseRow<float> >(rid, &row_acc);
      r.CopyToVector(&row_cache);
      uint32 msg_type = row_cache[kColIdxMsgType];
      if (msg_type == kProposal) {
        VIndex j = row_cache[kColIdxMsgVId];
        VIndex nbr_j = row_cache[kColIdxMsgNbrId];
        CIndex betap = row_cache[kColIdxMsgSt];
        CIndex nbrp = row_cache[kColIdxMsgSt + 1];
#ifdef DEBUG
        CHECK(vertices_.find(j) != vertices_.end());
        CHECK(neighbor_worker_.find(nbr_j) != neighbor_worker_.end());
#endif
        // feedback the msg
        MHSampleLinkFeedback(j, nbr_j, z_prev, betap, nbrp);

        ResetMsgRow(rid);    // reset the row 
        AdvanceIMsgRowId(w); // next msg
      } else if (msg_type == kInactive) { // empty msg, stop
        break;
      } else {
        LOG(FATAL) << "Illegial msg type " << msg_type;
      }
    } // end of rows
  }   // end of workers

  petuum::PSTableGroup::GlobalBarrier();

  /// Collect msgs and do accept-reject
  for (WIndex w = 0; w < num_workers_; ++w) {
    if (w == worker_id_) continue;
    uint32& rid = imsg_cur_rid_[w]
    while(true) {
       //TODO: set size to the maximum size ?
      vector<float> row_cache(kNumMsgPrfxCols + 1 + kNumMHStepStates);
      petuum::RowAccessor row_acc;
      const auto& r 
          = msg_table_.Get<petuum::DenseRow<float> >(rid, &row_acc);
      r.CopyToVector(&row_cache);
      uint32 msg_type = row_cache[kColIdxMsgType];
      if (msg_type == kFeedback) {
        VIndex i = row_cache[kColIdxMsgVId];
        VIndex j = row_cache[kColIdxMsgNbrId];
        CIndex jp = row_cache[kColIdxMsgSt];
#ifdef DEBUG
        CHECK(vertices_.find(i) != vertices_.end());
        CHECK(neighbor_worker_.find(j) != neighbor_worker_.end());
#endif
        // do accept-reject
        MHSampleAcptRjct(i, j, jp, &row_cache[kColIdxMsgSt+1]);

        ResetMsgRow(rid);    // reset the row 
        AdvanceIMsgRowId(w); // next msg
      } else if (msg_type == kInactive) { // empty msg, stop
        break;
      } else {
        LOG(FATAL) << "Illegial msg type " << msg_type;
      }
    } // end of rows
  }   // end of workers

  petuum::PSTableGroup::GlobalBarrier();
  /// update z from msg 
  SetZfromMsg();

  // temp
  //LOG(INFO) << "beta propsal accept ratio\t" 
  //    << (beta_proposal_accept_times_ * 1.0 / beta_proposal_times_) << "\t"
  //    << beta_proposal_accept_times_ << "\t" << beta_proposal_times_;
  //LOG(INFO) << "vertex propsal accept ratio\t" 
  //    << (vertex_proposal_accept_times_ * 1.0 / vertex_proposal_times_) << "\t"
  //    << vertex_proposal_accept_times_ << "\t" << vertex_proposal_times_;
}

void MMSBModel::MHSampleLinkRemote(
    const VIndex i, const VIndex j, const CIndex z_prev) {
#ifdef DEBUG
  CHECK(vertices_.find(i) != vertices_.end());
  CHECK(neighbor_worker_.find(j) != neighbor_worker_.end());
#endif
  CIndex s;
 
  /// beta proposal
  s = z_prev;
  CIndex betap = beta_alias_table_.Propose();

  /// prefetch 1 i-proposal
  Vertex* v_a = vertices_[i];
  CIndex ip;
  float nak_or_alpha = Context::rand() * (alpha_ * K_ + v_a->degree()); 
  if (nak_or_alpha < v_a->degree()) { // propose by n_ak
    uint32 nidx = Context::randUInt64() % v_a->degree();
    ip = v_a->z_by_neighbor_idx(nidx);
  } else { // propose by alpha (uniform)
    ip = Context::randUInt64() % K_;
  }
  /// store the 2 proposals locally
  pair<CIndex, CIndex>& betav_prpsls = v_a->nbr_betav_prpsls(j);
  betav_prpsls.first = betap;
  betav_prpsls.second = ip;
  /// send the 2 proposals to vertex j
  // row id
  WIndex nbr_w = neighbor_worker_[j];
  uint32 row_id = GetOMsgRowId(nbr_w);
  // msg content
  petuum::DenseUpdateBatch<float> update_batch(
      0, kNumMsgPrfxCols + 2);
  update_batch[kColIdxMsgType] = kProposal;
  update_batch[kColIdxMsgVId] = j;
  update_batch[kColIdxMsgNbrId] = i;
  update_batch[kColIdxMsgSt] = betap;
  update_batch[kColIdxMsgSt + 1] = ip;
  // send the msg 
  msg_table_.DenseBatchInc(row_id, update_batch);
}

void MMSBModel::MHSampleLinkFeedback(const VIndex j, const VIndex nbr_j,
    const CIndex z_prev, const CIndex betap, const CIndex nbrp) {
#ifdef DEBUG
  CHECK(vertices_.find(j) != vertices_.end());
  CHECK(neighbor_worker_.find(nbr_j) != neighbor_worker_.end());
#endif
  Vertex* v_a = vertices_[j];
/*
  /// prefetch 4 j-proposals
  CIndex jp[4];
  float jp_prob[4];
  for (int p=0; p<4; ++p) {
    float nak_or_alpha = Context::rand() * (alpha_ * K_ + v_a->degree()); 
    if (nak_or_alpha < v_a->degree()) { // propose by n_ak
      uint32 nidx = Context::randUInt64() % v_a->degree();
      jp[p] = v_a->z_by_neighbor_idx(nidx);
    } else { // propose by alpha (uniform)
      jp[p] = Context::randUInt64() % K_;
    }
  }
*/
  /// prefetch *1* j-proposal
  CIndex jp;
  float nak_or_alpha = Context::rand() * (alpha_ * K_ + v_a->degree()); 
  if (nak_or_alpha < v_a->degree()) { // propose by n_ak
    uint32 nidx = Context::randUInt64() % v_a->degree();
    jp = v_a->z_by_neighbor_idx(nidx);
  } else { // propose by alpha (uniform)
    jp = Context::randUInt64() % K_;
  }

  /// compute j-related terms of the reject-accept ratios
  float jp_prob[kNumMHStepStates];
  // *:  z_prev -> betap (ratio for beta-proposal) 
  jp_prob[kB] = ComputeMHRatioTerm(v_a, z_prev, betap, z_prev, kBetaPrpsl);
  // 0*:     z_prev -> nbrp (ratio for nrbj-proposal)
  jp_prob[kI0] = ComputeMHRatioTerm(v_a, z_prev, nbrp, z_prev, kViPrpsl);
  // 1*:     betap -> nbrp
  jp_prob[kI1] = ComputeMHRatioTerm(v_a, betap, nbrp, z_prev, kViPrpsl);
  // 00*:    z_prev -> jp (ratio for j-proposal)
  jp_prob[kJ00] = ComputeMHRatioTerm(v_a, z_prev, jp, z_prev, kVjPrpsl);
  // 10*:    betap -> jp
  jp_prob[kJ10] = ComputeMHRatioTerm(v_a, betap, jp, z_prev, kVjPrpsl);
  // [01]1*: nbrp -> jp
  jp_prob[kJw1] = ComputeMHRatioTerm(v_a, nbrp, jp, z_prev, kVjPrpsl);


  /// send the msg
  // row id
  WIndex nbr_w = neighbor_worker_[nbr_j];
  uint32 row_id = GetOMsgRowId(nbr_w);
  // msg content
  petuum::DenseUpdateBatch<float> update_batch(
      0, kNumMsgPrfxCols + 7);
  update_batch[kColIdxMsgType] = kFeedback;
  update_batch[kColIdxMsgVId] = nbr_j;
  update_batch[kColIdxMsgNbrId] = j;
  update_batch[kColIdxMsgSt] = jp;
  for (int p=0; p<kNumMHStepStates; ++p) {
    update_batch[kColIdxMsgSt + 1 + p] = jp_prob[p];
  }
  msg_table_.DenseBatchInc(row_id, update_batch);
}

float MMSBModel::ComputeMHRatioTerm(
    Vertex* v, CIndex s, CIndex t, CIndex z_prev, int mh_step) {
  float ns_alpha = v->z_cnt(s) + alpha_;
  float nt_alpha = v->z_cnt(t) + alpha_;
  //if (s != z_prev) {
  //  ns_alpha += 1.0;
  //  if (t == z_prev) {
  //    nt_alpha -= 1.0;
  //  }
  //}
  float ns_alpha_1 = ns_alpha; 
  float nt_alpha_1 = nt_alpha;
  if (s == z_prev) {
    ns_alpha_1 -= 1.0;
  }
  if (t == z_prev) {
    nt_alpha_1 -= 1.0;
  }
#ifdef DEBUG
  CHECK_GT(ns_alpha_1, 0);
  CHECK_GT(nt_alpha_1, 0);
#endif
  float ratio_term = 0;
  if (mh_step == kBetaPrpsl || mh_step == kViPrpsl) { // beta/nbr_j proposal
    ratio_term = nt_alpha_1 / ns_alpha_1;
  } else if (mh_step == kVjPrpsl) {         // j proposal
    ratio_term = nt_alpha_1 * ns_alpha / (ns_alpha_1 * nt_alpha);
  } else {
    LOG(FATAL) << "Illegal MHStep " << MHStep;
  }
  return ratio_term; 
}

void MHSampleAcptRjct(const VIndex i, const VIndex j,
    const CIndex jp, const float* ratio_jterms) {
#ifdef DEBUG
  CHECK(vertices_.find(i) != vertices_.end());
  CHECK(neighbor_worker_.find(j) != neighbor_worker_.end());
#endif
  Vertex* v_a = vertices_[i];
  const pair<CIndex, CIndex>& betav_prpsls = v_a->nbr_betav_prpsls(j);
  
  const CIndex z_prev = v_a->z_by_vertex_id(j);
  CIndex s = z_prev;
  CIndex t = betav_prpsls.first;
  /// accept-reject beta-proposal
  float nas_alpha = v_a->z_cnt(s) + alpha_;
  float nat_alpha = v_a->z_cnt(t) + alpha_;
  float nas_alpha_1 = nas_alpha;
  float nat_alpha_1 = nat_alpha;
  if (s == z_prev) {
    nas_alpha_1 -= 1.0;
  }
  if (t == z_prev) {
    nat_alpha_1 -= 1.0;
  }
#ifdef DEBUG
  CHECK_GT(nas_alpha_1, 0);
  CHECK_GT(nat_alpha_1, 0);
#endif

  float accept_ratio = min((float)1.0, 
      ratio_jterms[kB] * nat_alpha_1 / nas_alpha_1);
  bool b_accept = (Context::rand() < accept_ratio);
  s = b_baccept ? t : s;

  beta_proposal_times_++;
  beta_proposal_accept_times_ += (b_accept ? 0 : 1);

  /// accept-reject i-proposal
  t = betav_prpsls.second;
  nas_alpha = v_a->z_cnt(s) + alpha_;
  nat_alpha = v_a->z_cnt(t) + alpha_;
  nas_alpha_1 = nas_alpha;
  nat_alpha_1 = nat_alpha;
  if (s == z_prev) {
    nas_alpha_1 -= 1.0;
  }
  if (t == z_prev) {
    nat_alpha_1 -= 1.0;
  }
#ifdef DEBUG
  CHECK_GT(nas_alpha_1, 0);
  CHECK_GT(nat_alpha_1, 0);
#endif

  float jterm = b_accept ? ratio_jterms[kI1] : ratio_jterms[kI0];
  accept_ratio = min((float)1.0,
      jterm * (nat_alpha * exp_Elog_beta_[t] * nas_alpha_1) 
       / (nas_alpha * exp_Elog_beta_[s] * nat_alpha_1));
  float i_accept = (Context::rand() < accept_ratio);
  s = i_accept ? t : s;

  vertex_proposal_times_++;
  vertex_proposal_accept_times_ += (i_accept ? 0 : 1);

  /// accept-reject j-proposal
  t = jp;
  nas_alpha = v_a->z_cnt(s) + alpha_;
  nat_alpha = v_a->z_cnt(t) + alpha_;
  nas_alpha_1 = nas_alpha;
  nat_alpha_1 = nat_alpha;
  if (s == z_prev) {
    nas_alpha_1 -= 1.0;
  }
  if (t == z_prev) {
    nat_alpha_1 -= 1.0;
  }
#ifdef DEBUG
  CHECK_GT(nas_alpha_1, 0);
  CHECK_GT(nat_alpha_1, 0);
#endif

  float jterm = 0;
  if (i_accept) {
    jterm = ratio_jterms[kJw1];
  } else if (b_accept){
    jterm = ratio_jterms[kJ10];
  } else {
    jterm = ratio_jterm[kJ00];
  }
  accept_ratio = min((float)1.0,
      jterm * (exp_Elog_beta_[t] * nas_alpha_1) 
       / (exp_Elog_beta_[s] * nat_alpha_1));
  float j_accept = (Context::rand() < accept_ratio);
  s = j_accept ? t : s;
  
  vertex_proposal_times_++;
  vertex_proposal_accept_times_ += (j_accept ? 0 : 1);

  /// set Z
  if (s != z_prev) {
    v_a->RemoveZ(z_prev);
    v_a->SetZ(j, s);
    SendZMsg(j, i, s);
  }
}

CIndex MMSBModel::MHSampleLink(
    const Vertex* v_i, const Vertex* v_j, const CIndex z_prev) {
  CIndex s;

  /// beta proposal
  s = z_prev;
  CIndex t = beta_alias_table_.Propose();
  // accept-reject
  float nis_alpha = v_i->z_cnt(s) + alpha_;
  float nit_alpha = v_i->z_cnt(t) + alpha_;
  float njs_alpha = v_j->z_cnt(s) + alpha_;
  float njt_alpha = v_j->z_cnt(t) + alpha_;
  if (s == z_prev) {
    nis_alpha -= 1.0;
    njs_alpha -= 1.0;
  }
  if (t == z_prev) {
    nit_alpha -= 1.0;
    njt_alpha -= 1.0;
  }

  float accept_ratio = min((float)1.0, 
      (nit_alpha * njt_alpha) / (nis_alpha * njs_alpha));
  bool accept = (Context::rand() < accept_ratio);
  s = accept ? t : s;

  beta_proposal_times_++;
  beta_proposal_accept_times_ += (accept ? 0 : 1);

  /// i proposal
  s = MHSampleWithVertexProposal(v_i, v_j, z_prev, s);

  /// j proposal
  s = MHSampleWithVertexProposal(v_j, v_i, z_prev, s);

  return s;
}

/**
 * Propose the community of link (v_a, v_b) by v_a
 */
CIndex MMSBModel::MHSampleWithVertexProposal(
    const Vertex* v_a, const Vertex* v_b,
    const CIndex z_prev, const CIndex s) {
  CIndex t;
  float nak_or_alpha = Context::rand() * (alpha_ * K_ + v_a->degree()); 
  if (nak_or_alpha < v_a->degree()) { // propose by n_ak
    uint32 nidx = Context::randUInt64() % v_a->degree();
    t = v_a->z_by_neighbor_idx(nidx);
  } else { // propose by alpha (uniform)
    t = Context::randUInt64() % K_;
  }
  // accept-reject
  float nas_alpha = v_a->z_cnt(s) + alpha_;
  float nat_alpha = v_a->z_cnt(t) + alpha_;
  float nbs_alpha = v_b->z_cnt(s) + alpha_;
  float nbt_alpha = v_b->z_cnt(t) + alpha_;
  float nat_alpha_proposal = nat_alpha;
  float nas_alpha_proposal = nas_alpha;
  if (s == z_prev) {
    nas_alpha -= 1.0;
    nbs_alpha -= 1.0;
  }
  if (t == z_prev) {
    nat_alpha -= 1.0;
    nbt_alpha -= 1.0;
  }

  float accept_ratio = min((float)1.0,
      (nat_alpha * nbt_alpha * exp_Elog_beta_[t] * nas_alpha_proposal) 
       / (nas_alpha * nbs_alpha * exp_Elog_beta_[s] * nat_alpha_proposal));
  bool accept = (Context::rand() < accept_ratio);

  vertex_proposal_times_++;
  vertex_proposal_accept_times_ += (accept ? 0 : 1);

  return accept ? t : s;
}

void MMSBModel::VIComputeGrads() {
  //float batch_scale 
  //    = vertices_.size() * (vertices_.size() - 1.0) 
  //      / (train_batch_vertices_.size() * (train_batch_vertices_.size() - 1.0));
  //TODO: temp
  //batch_scale = 1.0;
  float batch_scale = 0.5 * vertices_.size() / train_batch_vertices_.size();

  // Reset grads & intermidiate stats
  fill(lambda_grads_.begin(), lambda_grads_.end(), make_pair(0, 0));
  fill(bphi_sum_.begin(), bphi_sum_.end(), 0);
  fill(bphi_square_sum_.begin(), bphi_square_sum_.end(), 0);

  for (const auto i : train_batch_vertices_) {
    CHECK_LT(i, vertices_.size());
    float degree = vertices_[i]->degree();
    if (degree == 0) {
      continue;
    }
    //CHECK_GT(degree, 0) << i;
    const unordered_map<CIndex, Count>& z_cnts = vertices_[i]->z_cnts();
    for (const auto z_cnt : z_cnts) {
      CHECK_LT(z_cnt.first, K_);
      bphi_sum_[z_cnt.first] += z_cnt.second / degree;
      bphi_square_sum_[z_cnt.first] 
          += (z_cnt.second / degree) * (z_cnt.second / degree);
    }
  } // end of vertices

  for (const auto& link : train_batch_links_) {
    VIndex i = link.first;
    VIndex j = link.second;
    const unordered_map<CIndex, Count>& z_cnts_i = vertices_[i]->z_cnts();
    Vertex* v_j = vertices_[j];
    float degree_i = vertices_[i]->degree();
    float degree_j = v_j->degree();
    //TODO(zhiting) iterates i or j whose degree is smaller
    for (const auto z_cnt_i : z_cnts_i) { 
      CIndex z = z_cnt_i.first;
      lambda_grads_[z].second
          -= (z_cnt_i.second / degree_i) * (v_j->z_cnt(z) / degree_j);
    }
  } // end of links

  for (int k = 0; k < K_; ++k) {
    lambda_grads_[k].first += 2.0 * train_batch_k_cnts_[k]; // !!Caution

    CHECK(!isnan(lambda_grads_[k].first)) << k << "\t" << train_batch_k_cnts_[k] << "\t" << batch_scale;
    CHECK(!isinf(lambda_grads_[k].first)) << k << "\t" << train_batch_k_cnts_[k] << "\t" << batch_scale;

    lambda_grads_[k].second
        += (bphi_sum_[k] * bphi_sum_[k] - bphi_square_sum_[k]) / 2.0;

    CHECK(!isnan(lambda_grads_[k].second)) << k << "\t" << bphi_sum_[k] << "\t" << bphi_square_sum_[k];
  } 
  for (int k = 0; k < K_; ++k) {
    float temp = lambda_grads_[k].first;
    lambda_grads_[k].first = lambda_grads_[k].first * batch_scale + eta_.first;

    CHECK(!isnan(lambda_grads_[k].first)) << k << "\t" << train_batch_k_cnts_[k] << "\t" << temp << "\t" << batch_scale;
    CHECK(!isinf(lambda_grads_[k].first)) << k << "\t" << train_batch_k_cnts_[k] << "\t" << temp << "\t" << batch_scale;

    temp = lambda_grads_[k].second;
    lambda_grads_[k].second = lambda_grads_[k].second * batch_scale + eta_.second;

    CHECK(!isnan(lambda_grads_[k].second)) << k << "\t" << temp << "\t" << batch_scale;
    CHECK(!isinf(lambda_grads_[k].second)) << k << "\t" << temp << "\t" << batch_scale;
  }
}

void MMSBModel::VIUpdate() {
  /// push updates to PS
  petuum::DenseUpdateBatch<float> update_batch_0(0, K_);
  petuum::DenseUpdateBatch<float> update_batch_1(0, K_);
  float lr = lr_;
  for (int k = 0; k < K_; ++k) {
    update_batch_0[k]
        = lr * (lambda_grads_[k].first - lambda_0_[k]);
    update_batch_1[k]
        = lr * (lambda_grads_[k].second - lambda_1_[k]);
  }
  // send the msg 
  msg_table_.DenseBatchInc(0, update_batch_0);
  msg_table_.DenseBatchInc(1, update_batch_1);

  /// fetch the latest values
  petuum::RowAccessor row_acc;
  const auto& r0
      = param_table_.Get<petuum::DenseRow<float> >(0, &row_acc);
  r0.CopyToVector(&lambda_0_);
  const auto& r1
      = param_table_.Get<petuum::DenseRow<float> >(1, &row_acc);
  r1.CopyToVector(&lambda_1_);
}

void MMSBModel::EstimateBeta() {
  ostringstream oss;
  for (int k = 0; k < K_; ++k) {
    beta_[k] = lambda_0_[k] / (lambda_0_[k] + lambda_1_[k]);
    oss << beta_[k] << "\t";
  }
  //LOG(INFO) << oss.str();
}

void MMSBModel::ComputeExpELogBeta() { 
  for (int k = 0; k < K_; ++k) {
    exp_Elog_beta_[k]
        = std::exp(mmsb::digamma(lambda_0_[k])
        - mmsb::digamma(lambda_0_[k].first + lambda_1_[k]));
  }
}

/// Assume beta_ is ready
float MMSBModel::ComputeLinkLikelihood(
    const VIndex i, const VIndex j, const bool positive) {
#ifdef DEBUG
  CHECK(vertices_.find(i) != vertices_.end());
  CHECK(vertices_.find(j) != vertices_.end());
#endif
  float likelihood = 0;
  Vertex* v_i = vertices_[i];
  Vertex* v_j = vertices_[j];
  if (positive) {
    for (int z = 0; z < K_; ++z) {
      likelihood += MembershipProb(v_i->z_cnt(z), v_i->degree())
          * MembershipProb(v_j->z_cnt(z), v_j->degree())
          * beta_[z];
    }
  } else {
    for (int zi = 0; zi < K_; ++zi) {
      for (int zj = 0; zj < K_; ++zj) {
        float bernu_rate = (zi == zj ? beta_[zi] : epsilon_);
        likelihood += MembershipProb(v_i->z_cnt(zi), v_i->degree())
            * MembershipProb(v_j->z_cnt(zj), v_j->degree())
            * (1.0 - bernu_rate);
      }
    }
  }
  CHECK(!isnan(likelihood));
  CHECK(!isinf(likelihood));
  CHECK_GT(likelihood, 0);
  return log(likelihood);
}

float MMSBModel::ComputeLinkLikelihoodRemote(
    const VIndex i, const VIndex j, const bool positive) {
#ifdef DEBUG
  CHECK(test_neighbor_worker_.find(j) != neighbor_worker__.end());
#endif
  Vertex* v_i = vertices_[i];
  const auto& z_cnts = v_i->z_cnts();
  WIndex nbr_w = test_neighbor_worker_[j];
  const uint32 rid = GetOMsgRowId(nbr_w);
  // msg content
  petuum::DenseUpdateBatch<float> update_batch(
      0, kNumMsgPrfxCols + z_cnts.size() * 2);
  update_batch[kColIdxMsgType] = kSetZ;
  update_batch[kColIdxMsgVId] = nbr_i;
  update_batch[kColIdxMsgNbrId] = i;
  update_batch[kColIdxMsgSt] = z;
  // send the msg 
  msg_table_.DenseBatchInc(row_id, update_batch);
   
}

void MMSBModel::Test() {
  //LOG(INFO) << "Testing ... ";
  clock_t test_start_time = std::clock();

  EstimateBeta();

  float lld = 0, lld_pos = 0, lld_neg = 0; // log likelihood
  Count num_links = 0, num_pos_links = 0, num_neg_links = 0;
  for (int t=0; t<test_links_.size(); ++t) {
    const auto& link = test_links_[t];
    VIndex i = link.first;
    VIndex j = link.second;
    const int value = test_link_values_[t];
#ifdef DEBUG
    CHECK(vertices_.find(i) != vertices_.end());
    CHECK(vertices_.find(j) != vertices_.end() ||
        test_neibor_worker_.find(j) != test_neibor_worker_.end());
#endif
    if (vertices_.find(j) != vertices_.end()) { 
      float lld_ij = ComputeLinkLikelihood(i, j, value);
      lld += lld_ij;
      if (value) {
        lld_pos += lld_ij;
      } else {
        lld_neg += lld_ij;
      }
      link_cnt++;
    } else {

    }
  }

  start_time_ += (std::clock() - test_start_time); // don't account for test time
  float duration = (std::clock() - start_time_) / (float)CLOCKS_PER_SEC;  
  LOG(ERROR) << iter_ << "\t" << lld / link_cnt << "\t" << link_cnt 
      << "\t" << (lld_pos / test_pos_links_.size()) 
      << "\t" << (lld_neg / test_neg_links_.size())
      << "\t" << duration;
}

void MMSBModel::Solve(const char* resume_file) {
  LOG(INFO) << "Solving MMSB";

  iter_ = 0;
  //if (resume_file) {
  //  LOG(INFO) << "Restoring previous model status from " << resume_file;
  //  Restore(resume_file);
  //  LOG(INFO) << "Restoration done.";
  //} else {
    InitModelState();
  //}

  const int start_iter = iter_;

  start_time_ = std::clock();
  for (; iter_ < param_.solver_param().max_iter(); ++iter_) {
    /// Save a snapshot if needed.
    //if (param_.solver_param().snapshot() && iter_ > start_iter &&
    //    iter_ % param_.solver_param().snapshot() == 0) {
    //  Snapshot();
    //  SnapshotVis();
    //}

    /// Test if needed
    //if (param_.solver_param().test_interval() 
    //    && iter_ % param_.solver_param().test_interval() == 0
    //    && (iter_ > 0 || param_.solver_param().test_initialization())) {
    //  //clock_t start_test = std::clock();
    //  Test();
    //  //LOG(INFO) << "test time " << (std::clock() - start_test) / (float)CLOCKS_PER_SEC;
    //}

    /// Sample mini-batch
    //clock_t start_data = std::clock();
    SampleMinibatch(train_batch_vertices_, train_batch_links_, train_batch_size_);
    //LOG(INFO) << "data time " << (std::clock() - start_data) / (float)CLOCKS_PER_SEC;

    /// local step
    //clock_t start_sample = std::clock();
    if (param_.solver_param().sampler_type() == mmsb::SamplerType::MHSampler) {
      MHSample(); // O(1) sampler
    } else {
      GibbsSample(); // O(K) sampler
    }
    //LOG(INFO) << "sample time " << (std::clock() - start_sample) / (float)CLOCKS_PER_SEC;

    /// global step
    //clock_t start_vi = std::clock();
    VIComputeGrads();
    VIUpdate();
    //LOG(INFO) << "vi time " << (std::clock() - start_vi) / (float)CLOCKS_PER_SEC;
  }
  if (param_.solver_param().snapshot_after_train()) {
    Snapshot(); 
  }
  if (param_.solver_param().test_interval() &&
      iter_ % param_.solver_param().test_interval() == 0) {
    Test();
  }
}

void MMSBModel::ToProto(ModelParameter* param) {
  param->Clear();
  // hyperparams
  param->CopyFrom(param_);
  param->clear_vertices();
  // global params (lambda)
  param->clear_lambda_0();  
  param->clear_lambda_1();
  for (CIndex k = 0; k < K_; ++k) {
    param->add_lambda_0(lambda_0_[k]);
    param->add_lambda_1(lambda_1_[k]);
  }
  // vertex's params (neighbors, z)
  for (VIndex i = 0; i < vertices_.size(); ++i) {
    Vertex* v = vertices_[i];
    VertexParameter* v_param = param->add_vertices();
    v->ToProto(v_param);
    v_param->set_index(i);
  }
  // solver state
  param->mutable_solver_param()->mutable_solver_state()->set_iter(iter_);
}

void MMSBModel::Snapshot() {
  const string& output_prefix = Context::get_string("mmsb_output");
  ostringstream oss;
  oss << output_prefix << "mmsb.iter." << iter_; 
  string snapshot_filename = oss.str();

  ModelParameter model_param;
  ToProto(&model_param);
  LOG(INFO) << "Snapshotting to " << snapshot_filename;

  WriteProtoToBinaryFile(model_param, snapshot_filename.c_str());
}

void MMSBModel::SnapshotVis() {
  const string& output_prefix = Context::get_string("mmsb_output");
  LOG(INFO) << "Snapshoting (Vis) to " << output_prefix;
  // community strengths: lambda, beta
  EstimateBeta();
  ostringstream oss;
  oss << output_prefix + "/comm_strength_" << iter_ << ".txt"; 
  string lambda_path = oss.str();
  fstream lambda_file(lambda_path.c_str(), ios::out);
  CHECK(lambda_file.is_open()) << "Fail to open " << lambda_path;
  for (int k = 0; k < K_; ++k) {
    lambda_file << k << "\t" << lambda_0_[k] << "\t" << lambda_1_[k]
        << "\t" << beta_[k] << "\n";
  }
  lambda_file.flush();
  lambda_file.close();

  // user membership: z_cnts
  oss.str("");
  oss.clear();
  oss << output_prefix + "/user_membership_" << iter_ << ".txt"; 
  string z_path = oss.str();
  fstream z_file(z_path.c_str(), ios::out);
  CHECK(z_file.is_open()) << "Fail to open " << z_path;
  for (int i = 0; i < vertices_.size(); ++i) {
    z_file << i << "\t" << vertices_[i]->degree() << "\t";
    const unordered_map<CIndex, Count>& z_cnts = vertices_[i]->z_cnts();
    for (const auto& e : z_cnts) {
      z_file << e.first << ":" << e.second << "\t"; 
    }
    z_file << "\n";
  }
  z_file.flush();
  z_file.close();

  LOG(INFO) << "Snapshot (Vis) done.";
}

void MMSBModel::Restore(const char* snapshot_file) {
  ModelParameter param;
  ReadProtoFromBinaryFile(snapshot_file, &param);
  // check consistency
  CHECK_EQ(K_, param.num_comms());
  CHECK_EQ(vertices_.size(), param.vertices_size());
  // global param
  for (CIndex k = 0; k < K_; ++k) {
    lambda_0_[k] = param.lambda_0(k);
    lambda_1_[k] = param.lambda_1(k);
  }
  // vertex's params (neighbors, z)
  for (VIndex i = 0; i < vertices_.size(); ++i) {
    const VertexParameter& v_param = param.vertices(i);
    CHECK_EQ(i, v_param.index());
    vertices_[i]->FromProto(v_param);
  }
  // solver state
  iter_ = param.solver_param().solver_state().iter();
}

} // namespace mmsb
