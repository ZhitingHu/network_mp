
#include "common.hpp"
#include "context.hpp"
#include "mmsb_engine.hpp"
#include "mmsb_model.hpp"

namespace mmsb {

MMSBEngine::MMSBEngine() : thread_counter_(0) {

}

void MMSBEngine::Init() {
  const string& model_path = Context::get_string("model");
  ReadProtoFromTextFile(param_file, &param_);

  LOG(INFO) << "Init PS environment";
  petuum::TableGroupConfig table_group_config;
  table_group_config.num_comm_channels_per_client
      = Context::get_int32("num_comm_channels_per_client");
  table_group_config.num_total_clients 
      = Context::get_int32("num_clients");
  // + 1 for main() thread.
  table_group_config.num_local_app_threads 
      = Context::get_int32("num_app_threads") + 1;
  table_group_config.client_id = Context::get_int32("client_id");
  table_group_config.stats_path = Context::get_string("stats_path");
  petuum::GetHostInfos(Context::get_string("hostfile"), 
      &table_group_config.host_map);
  string consistency_model = Context::get_string("consistency_model");
  if (std::string("SSP").compare(consistency_model) == 0) {
    table_group_config.consistency_model = petuum::SSP;
  } else if (
    std::string("SSPPush").compare(consistency_model) == 0) {
    table_group_config.consistency_model = petuum::SSPPush;
  } else {
    LOG(FATAL) << "Unkown consistency model: " << consistency_model;
  }
  // msg_table & param_table
  table_group_config.num_tables = kNumPSTables;
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float> >
    (kFloatDenseRowDtypeID);

  // Use false to not let main thread access table API.
  petuum::PSTableGroup::Init(table_group_config, false);
  LOG(INFO) << "Init table group done.";

  CreateTables();
}

void MMSBEngine::CreateTables() {
  LOG(INFO) << "Create tables.";
  int msg_table_staleness = Context::get_int32("msg_table_staleness");
  int param_table_staleness = Context::get_int32("param_table_staleness");
  int row_oplog_type = Context::get_int32("row_oplog_type");
  bool oplog_dense_serialized = Context::get_bool("oplog_dense_serialized");
  int num_threads = Context::get_int32("num_app_threads");
  int tot_num_threads = Context::get_int32("num_clients") * num_threads;
  // common table config
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_oplog_type = row_oplog_type;
  table_config.table_info.oplog_dense_serialized 
      = oplog_dense_serialized;
  table_config.process_storage_type = petuum::BoundedSparse;

  /// message table
  // the maximum possible length of a msg
  uint32 msg_row_length = kNumMsgPrfxCols+1+kNumMsgPrfxCols; 
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = msg_table_staleness;
  table_config.table_info.row_capacity = msg_row_length;
  table_config.process_cache_capacity = kMsgTabMaxNumRows / tot_num_threads + 1; //TODO
  table_config.table_info.dense_row_oplog_capacity = msg_row_length + 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kMsgTableID, table_config);
  LOG(INFO) << "Created msg table " << 0;

  /// param table
  uint32 param_row_length = param_.num_comms(); 
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = param_table_staleness;
  table_config.table_info.row_capacity = param_row_length;
  table_config.process_cache_capacity = 2; //TODO
  table_config.table_info.dense_row_oplog_capacity = param_row_length + 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kParamTableID, table_config);
  LOG(INFO) << "Created param table " << 0;

  petuum::PSTableGroup::CreateTableDone(); 
}

void MMSBEngine::Start() {
  petuum::PSTableGroup::RegisterThread();

  // Initialize local thread data structures.
  int thread_id = thread_counter_++;
  int client_id = Context::get_int32("client_id");

  const string& snapshot_path = Context::get_string("snapshot");

  MMSBModel* model = new MMSBModel(param_, client_id, thread_id);
  model->ReadData();

  if (snapshot_path.size()) {
    LOG(INFO) << "Resuming from " << snapshot_path;
    model->Solve(snapshot_path);
  } else {
    model->Solve();
  }

  petuum::PSTableGroup::GlobalBarrier();
  petuum::PSTableGroup::DeregisterThread();
}

} // namespace mmsb
