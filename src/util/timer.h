#ifndef __MY_TIMER_H_
#define __MY_TIMER_H_

#include <chrono>

namespace T {
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::time_point;

void Init();
void Start(int i);
void End(int i);
high_resolution_clock::duration Get(int i);
}  // namespace T

#endif
