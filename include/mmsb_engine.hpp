#ifndef MMSB_MMSB_ENGINE_HPP_
#define MMSB_MMSB_ENGINE_HPP_

#include "common.hpp"
#include "context.hpp"
#include <atomic>

namespace mmsb {

class MMSBEngine {
 public:
  explicit MMSBEngine();
  
  void Init();

  //void ReadData();

  // Can be called concurrently.  
  void Start();
 
 private:
  void CreateTables();

 private:

  ModelParameter param_;

  // 
  std::atomic<int> thread_counter_;

};

} // namespace mmsb 

#endif
