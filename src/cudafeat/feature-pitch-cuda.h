
#include "feat/pitch-functions.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

namespace kaldi {

class ComputePitchCuda {
 public:
  void ComputePitch(const CuVectorBase<BaseFloat> &cu_wave,
                    CuMatrix<BaseFloat> *cu_pitch_pov);

  ComputePitchCuda(const PitchExtractionOptions &opts);

private:
  PitchExtractionOptions opts;
};

class ProcessPitchCuda {
 public:
  void ProcessPitch(const CuMatrix<BaseFloat> &cu_pitch_pov,
                    CuMatrix<BaseFloat>* cu_features);

  ProcessPitchCuda(const ProcessPitchOptions &opts);
};

}  // namespace kaldi
