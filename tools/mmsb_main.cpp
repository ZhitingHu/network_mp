
#include "mmsb.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <petuum_ps_common/include/petuum_ps.hpp>

// MMSB Parameters
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot state to resume training.");
DEFINE_string(mmsb_output, "",
    "The prefix of the mmsb output file.");
// Data Parameters
DEFINE_string(train_data, "",
    "The training data path.");
DEFINE_string(test_data, "",
    "The test data path.");
//Other Parameters
DEFINE_int32(random_seed, -1,
    "Use system time as rand seed by default.");

// Petuum Parameters
DEFINE_string(hostfile, "",
    "Path to file containing server ip:port.");
DEFINE_int32(num_clients, 1, 
    "Total number of clients");
DEFINE_int32(num_app_threads, 1, 
    "Number of app threads in this client");
DEFINE_int32(client_id, 0, 
    "Client ID");
DEFINE_string(consistency_model, "SSPPush", 
    "SSP or SSPPush");
DEFINE_string(stats_path, "", 
    "Statistics output file");
DEFINE_int32(num_comm_channels_per_client, 1,
    "number of comm channels per client");
DEFINE_int32(msg_table_staleness, 0, 
    "staleness of message tables.");
DEFINE_int32(param_table_staleness, 0, 
    "staleness of param tables.");
DEFINE_int32(row_oplog_type, petuum::RowOpLogType::kDenseRowOpLog,
    "row oplog type");
DEFINE_bool(oplog_dense_serialized, true, 
    "True to not squeeze out the 0's in dense oplog.");

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Do not buffer
  FLAGS_logbuflevel = -1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to train.";

  // Initialize Context
  mmsb::Context& context = mmsb::Context::Get(); 
  context.Init();

  mmsb::MMSBEngine* mmsb_engine = new mmsb::MMSBEngine();
  mmsb_engine->Init();

  LOG(INFO) << "Starting MMSB with " << FLAGS_num_app_threads << " threads "
      << "on client " << FLAGS_client_id;

  std::vector<std::thread> threads(FLAGS_num_app_threads); 
  for (auto& thr : threads) {
    thr = std::thread(&mmsb::MMSBEngine::Start, std::ref(*mmsb_engine));
  }
  for (auto& thr : threads) {
    thr.join();
  }

  LOG(INFO) << "Optimization Done.";

  petuum::PSTableGroup::ShutDown();
  LOG(INFO) << "MMSB finished and shut down!";

  return 0;
}
