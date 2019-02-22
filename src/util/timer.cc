#include "util/timer.h"
#include <vector>

namespace T {

static high_resolution_clock* timer;
static std::vector<high_resolution_clock::time_point>* start_time;
static std::vector<high_resolution_clock::duration>* duration;

void Init() {
  timer = new high_resolution_clock();
  start_time = new std::vector<high_resolution_clock::time_point>(100);
  duration = new std::vector<high_resolution_clock::duration>(100);
}
void Start(int i) { (*start_time)[i] = (*timer).now(); }
void End(int i) {
  auto end = (*timer).now();
  auto d = end - (*start_time)[i];
  (*duration)[i] += d;
}

high_resolution_clock::duration Get(int i) { return (*duration)[i]; }

}  // namespace T
