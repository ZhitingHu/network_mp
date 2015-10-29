#ifndef MMSB_COMMON_HPP_
#define MMSB_COMMON_HPP_

#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <limits>
#include <algorithm>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "proto/mmsb.pb.h"

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFAGS_H_ to detect if it is version
// 2.1. If yes , we will add a temporary solution to redirect the namespace.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace mmsb {

using std::fstream;
using std::ios;
using std::isnan;
using std::iterator;
using std::make_pair;
using std::vector;
using std::map;
using std::unordered_map;
using std::ostringstream;
using std::pair;
using std::set;
using std::unordered_set;
using std::string;
using std::stringstream;
using std::max;
using std::min;

// Typedefs
typedef unsigned short uint16; // Should work for all x86/x64 compilers
typedef unsigned int uint32; // Should work for all x86/x64 compilers
typedef uint32 VIndex; // Vertex index
typedef uint32 CIndex; // Community index
typedef uint32 Count;  // count
typedef uint32 WIndex; // Worker index 

const int kNumPSTables = 2;
enum TableIds {
  kMsgTableID = 0,
  kParamTableID
};

const uint32 kMsgTabMaxNumRows = 1e8; // max: 2^31 ~ 1e9
enum MsgType {
  kInactive = 0,
  kProposal,
  kFeedback,
  kSetZ,
  kTest
};

const int kNumMsgPrfxCols = 3;
enum MsgTableCols {
  kColIdxMsgType = 0,
  kColIdxMsgVId,
  kColIdxMsgNbrId
  kColIdxMsgSt
};

enum MHStep {
  kBetaPrpsl = 0,
  kViPrpsl,
  kVjPrpsl
}

const int kNumMHStepStates = 6;
enum MHStepStates {
  kB = 0, // beta-proposal
  kI0,    // i-proposal, 0*
  kI1,    // i-proposal, 1*
  kJ00,   // j-proposal, 00*
  kJ10,   // j-proposal, 10*
  kJw1,   // j-proposal, [01]1*
}

} // namespace mmsb

#endif
