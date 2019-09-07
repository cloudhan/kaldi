#include "cudafeat/feature-pitch-cuda.h"
#include "feat/resample.h"

namespace kaldi {

void WeightedMovingWindowNormalize(int32 normalization_window_size,
                                   const VectorBase<BaseFloat> &pov,
                                   const VectorBase<BaseFloat> &raw_log_pitch,
                                   Vector<BaseFloat> *normalized_log_pitch) {
  int32 num_frames = pov.Dim();
  KALDI_ASSERT(num_frames == raw_log_pitch.Dim());
  int32 last_window_start = -1, last_window_end = -1;
  double weighted_sum = 0.0, pov_sum = 0.0;

  for (int32 t = 0; t < num_frames; t++) {
    int32 window_start, window_end;
    window_start = t - (normalization_window_size / 2);
    window_end = window_start + normalization_window_size;

    if (window_start < 0) {
      window_end -= window_start;
      window_start = 0;
    }

    if (window_end > num_frames) {
      window_start -= (window_end - num_frames);
      window_end = num_frames;
      if (window_start < 0) window_start = 0;
    }
    if (last_window_start == -1) {
      SubVector<BaseFloat> pitch_part(raw_log_pitch, window_start,
                                      window_end - window_start);
      SubVector<BaseFloat> pov_part(pov, window_start,
                                    window_end - window_start);
      // weighted sum of pitch
      weighted_sum += VecVec(pitch_part, pov_part);

      // sum of pov
      pov_sum = pov_part.Sum();
    } else {
      if (window_start > last_window_start) {
        KALDI_ASSERT(window_start == last_window_start + 1);
        pov_sum -= pov(last_window_start);
        weighted_sum -=
            pov(last_window_start) * raw_log_pitch(last_window_start);
      }
      if (window_end > last_window_end) {
        KALDI_ASSERT(window_end == last_window_end + 1);
        pov_sum += pov(last_window_end);
        weighted_sum += pov(last_window_end) * raw_log_pitch(last_window_end);
      }
    }

    KALDI_ASSERT(window_end - window_start > 0);
    last_window_start = window_start;
    last_window_end = window_end;
    (*normalized_log_pitch)(t) = raw_log_pitch(t) - weighted_sum / pov_sum;
    KALDI_ASSERT((*normalized_log_pitch)(t) - (*normalized_log_pitch)(t) == 0);
  }
}

BaseFloat NccfToPovFeature(BaseFloat n) {
  if (n > 1.0) {
    n = 1.0;
  } else if (n < -1.0) {
    n = -1.0;
  }
  BaseFloat f = pow((1.0001 - n), 0.15) - 1.0;
  KALDI_ASSERT(f - f == 0);  // check for NaN,inf.
  return f;
}

BaseFloat NccfToPov(BaseFloat n) {
  BaseFloat ndash = fabs(n);
  if (ndash > 1.0) ndash = 1.0;  // just in case it was slightly outside [-1, 1]

  BaseFloat r = -5.2 + 5.4 * exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
                2.0 * exp(-10.0 * ndash) + 4.2 * exp(20.0 * (ndash - 1.0));
  // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
  BaseFloat p = 1.0 / (1 + exp(-1.0 * r));
  KALDI_ASSERT(p - p == 0);  // Check for NaN/inf
  return p;
}

int32 PitchNumFrames(int32 nsamp, const PitchExtractionOptions &opts) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  if (nsamp < frame_length)
    return 0;
  else {
    if (!opts.snip_edges) {
      return static_cast<int32>(nsamp * 1.0f / frame_shift + 0.5f);
    } else {
      return static_cast<int32>((nsamp - frame_length) / frame_shift + 1);
    }
  }
}

void PreemphasizeFrame(VectorBase<BaseFloat> *waveform, double preemph_coeff) {
  if (preemph_coeff == 0.0) return;
  KALDI_ASSERT(preemph_coeff >= 0.0 && preemph_coeff <= 1.0);
  for (int32 i = waveform->Dim() - 1; i > 0; i--)
    (*waveform)(i) -= preemph_coeff * (*waveform)(i - 1);
  (*waveform)(0) -= preemph_coeff * (*waveform)(0);
}

void ExtractFrame(const VectorBase<BaseFloat> &wave, int32 frame_index,
                  const PitchExtractionOptions &opts,
                  Vector<BaseFloat> *window) {
  int32 frame_shift = opts.NccfWindowShift();
  int32 frame_length = opts.NccfWindowSize();
  double outer_max_lag = 1.0 / opts.min_f0 + (opts.upsample_filter_width /
                                              (2.0 * opts.resample_freq));
  int32 end_lag = floor(opts.resample_freq * outer_max_lag);

  KALDI_ASSERT(frame_shift != 0 && frame_length != 0);
  int32 start = frame_shift * frame_index;
  int32 frame_length_full = frame_length + end_lag;

  if (window->Dim() != frame_length_full) window->Resize(frame_length_full);

  if (start + frame_length_full <= wave.Dim()) {
    window->CopyFromVec(wave.Range(start, frame_length_full));
  } else {
    window->SetZero();
    int32 size = wave.Dim() - start;
    window->Range(0, size).CopyFromVec(wave.Range(start, size));
  }

  if (opts.preemph_coeff != 0.0) PreemphasizeFrame(window, opts.preemph_coeff);
}

void PreProcess(const PitchExtractionOptions opts,
                const VectorBase<BaseFloat> &wave,
                Vector<BaseFloat> *processed_wave) {
  KALDI_ASSERT(processed_wave != NULL);
  // down-sample and Low-Pass filtering the input wave
  int32 num_samples_in = wave.Dim();
  double dt = opts.samp_freq / opts.resample_freq;
  int32 resampled_len = 1 + static_cast<int>(num_samples_in / dt);
  processed_wave->Resize(resampled_len);  // filtered wave
  LinearResample resample(opts.samp_freq, opts.resample_freq,
                          opts.lowpass_cutoff, opts.lowpass_filter_width);
  const bool flush = true;
  resample.Resample(wave, flush, processed_wave);
}

void ComputeCorrelation(const Vector<BaseFloat> &wave, int32 start, int32 end,
                        int32 nccf_window_size, Vector<BaseFloat> *inner_prod,
                        Vector<BaseFloat> *norm_prod) {
  Vector<BaseFloat> zero_mean_wave(wave);
  SubVector<BaseFloat> wave_part(wave, 0, nccf_window_size);
  // subtract mean-frame from wave
  zero_mean_wave.Add(-wave_part.Sum() / nccf_window_size);
  BaseFloat e1, e2, sum;
  SubVector<BaseFloat> sub_vec1(zero_mean_wave, 0, nccf_window_size);
  e1 = VecVec(sub_vec1, sub_vec1);
  for (int32 lag = start; lag <= end; lag++) {
    SubVector<BaseFloat> sub_vec2(zero_mean_wave, lag, nccf_window_size);
    e2 = VecVec(sub_vec2, sub_vec2);
    sum = VecVec(sub_vec1, sub_vec2);
    (*inner_prod)(lag - start) = sum;
    (*norm_prod)(lag - start) = e1 * e2;
  }
}

void ProcessNccf(const Vector<BaseFloat> &inner_prod,
                 const Vector<BaseFloat> &norm_prod, BaseFloat nccf_ballast,
                 SubVector<BaseFloat> *nccf_vec) {
  KALDI_ASSERT(inner_prod.Dim() == norm_prod.Dim() &&
               inner_prod.Dim() == nccf_vec->Dim());
  for (int32 lag = 0; lag < inner_prod.Dim(); lag++) {
    BaseFloat numerator = inner_prod(lag),
              denominator = pow(norm_prod(lag) + nccf_ballast, 0.5), nccf;
    if (denominator != 0.0) {
      nccf = numerator / denominator;
    } else {
      KALDI_ASSERT(numerator == 0.0);
      nccf = 0.0;
    }
    KALDI_ASSERT(nccf < 1.01 && nccf > -1.01);
    (*nccf_vec)(lag) = nccf;
  }
}

void SelectLags(const PitchExtractionOptions &opts, Vector<BaseFloat> *lags) {
  // choose lags relative to acceptable pitch tolerance
  BaseFloat min_lag = 1.0 / opts.max_f0, max_lag = 1.0 / opts.min_f0;

  std::vector<BaseFloat> tmp_lags;
  for (BaseFloat lag = min_lag; lag <= max_lag; lag *= 1.0 + opts.delta_pitch)
    tmp_lags.push_back(lag);
  lags->Resize(tmp_lags.size());
  std::copy(tmp_lags.begin(), tmp_lags.end(), lags->Data());
}

class PitchExtractor {
 public:
  PitchExtractor(const PitchExtractionOptions &opts,
                 const Vector<BaseFloat> &lags, int32 num_states,
                 int32 num_frames)
      : opts_(opts),
        num_states_(num_states),
        num_frames_(num_frames),
        lags_(lags) {
    frames_.resize(num_frames_ + 1);
    for (int32 i = 0; i < num_frames_ + 1; i++) {
      frames_[i].obj_func.Resize(num_states_);
      frames_[i].back_pointers.resize(num_states_);
    }
  }
  ~PitchExtractor() {}

  void ComputeLocalCost(const VectorBase<BaseFloat> &nccf_row,
                        VectorBase<BaseFloat> *local_cost) {
    // compute the local cost
    local_cost->Set(1.0);
    local_cost->AddVec(-1.0, nccf_row);
    Vector<BaseFloat> corr_lag_cost(num_states_);
    corr_lag_cost.AddVecVec(opts_.soft_min_f0, nccf_row, lags_, 0);
    local_cost->AddVec(1.0, corr_lag_cost);
  }

  void FastViterbi(const Matrix<BaseFloat> &nccf) {
    BaseFloat intercost, min_c, this_c;
    int best_b, min_i, max_i;
    const BaseFloat delta_pitch_sq = pow(log(1 + opts_.delta_pitch), 2.0);
    const BaseFloat inter_frame_factor = delta_pitch_sq * opts_.penalty_factor;

    // loop over frames
    for (int32 t = 1; t < num_frames_ + 1; t++) {
      // The stuff with the "forward pass" and "backward "pass" is described in
      // the paper; it's an algorithm that allows us to compute the vector of
      // forward-costs and back-pointers without accessing all of the pairs of
      // [pitch on last frame, pitch on this frame].
      Vector<BaseFloat> local_cost(num_states_);
      ComputeLocalCost(nccf.Row(t - 1), &local_cost);
      // std::cerr << local_cost << std::endl;

      // Forward Pass
      for (int32 i = 0; i < num_states_; i++) {
        if (i == 0)
          min_i = 0;
        else
          min_i = frames_[t].back_pointers[i - 1];
        min_c = std::numeric_limits<double>::infinity();
        best_b = -1;

        for (int32 k = min_i; k <= i; k++) {
          intercost = (i - k) * (i - k) * inter_frame_factor;
          this_c = frames_[t - 1].obj_func(k) + intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].back_pointers[i] = best_b;
        frames_[t].obj_func(i) = min_c + local_cost(i);
      }
      // Backward Pass
      for (int32 i = num_states_ - 1; i >= 0; i--) {
        if (i == num_states_ - 1)
          max_i = num_states_ - 1;
        else
          max_i = frames_[t].back_pointers[i + 1];
        min_c = frames_[t].obj_func(i) - local_cost(i);
        best_b = frames_[t].back_pointers[i];

        for (int32 k = i + 1; k <= max_i; k++) {
          intercost = (i - k) * (i - k) * inter_frame_factor;
          this_c = frames_[t - 1].obj_func(k) + intercost;
          if (this_c < min_c) {
            min_c = this_c;
            best_b = k;
          }
        }
        frames_[t].back_pointers[i] = best_b;
        frames_[t].obj_func(i) = min_c + local_cost(i);
      }

      // std::cerr << frames_[t].obj_func << std::endl;
      double remainder = frames_[t].obj_func.Min();
      frames_[t].obj_func.Add(-remainder);
    }

    Matrix<BaseFloat> cost(num_frames_, num_states_);
    for (int i = 0; i < num_frames_; i++) {
      SubVector<BaseFloat> cost_row(cost, i);
      cost_row.CopyFromVec(frames_[i + 1].obj_func);
    }
  }

  void FindBestPath(const Matrix<BaseFloat> &correlation) {
    // Find the Best path using backpointers
    int32 i = num_frames_;
    int32 best;
    double l_opt;
    frames_[i].obj_func.Min(&best);
    while (i > 0) {
      l_opt = lags_(best);
      frames_[i].truepitch = 1.0 / l_opt;
      frames_[i].pov = correlation(i - 1, best);
      best = frames_[i].back_pointers[best];
      i--;
    }
  }
  void GetPitchAndPov(Matrix<BaseFloat> *output) {
    output->Resize(num_frames_, 2);
    for (int32 frm = 0; frm < num_frames_; frm++) {
      (*output)(frm, 0) = static_cast<BaseFloat>(frames_[frm + 1].pov);
      (*output)(frm, 1) = static_cast<BaseFloat>(frames_[frm + 1].truepitch);
    }
  }

 private:
  PitchExtractionOptions opts_;
  int32 num_states_;        // number of states in Viterbi Computation
  int32 num_frames_;        // number of frames in input wave
  Vector<BaseFloat> lags_;  // all lags used in viterbi
  struct PitchFrame {
    Vector<BaseFloat> obj_func;  // optimal objective function for frame i
    std::vector<int32> back_pointers;
    BaseFloat truepitch;  // True pitch
    BaseFloat pov;        // probability of voicing
    explicit PitchFrame() {}
  };
  std::vector<PitchFrame> frames_;
};

void ComputePitchCuda::ComputePitch(const CuVectorBase<BaseFloat> &cu_wave,
                                    CuMatrix<BaseFloat> *cu_pitch_pov) {
  KALDI_ASSERT(cu_pitch_pov != NULL);

  // FIXME: remove copy
  Vector<BaseFloat> wave(cu_wave);

  // Preprocess the wave
  Vector<BaseFloat> processed_wave(wave.Dim());
  PreProcess(opts, wave, &processed_wave);

  int32 num_frames = PitchNumFrames(processed_wave.Dim(), opts);
  if (num_frames == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << processed_wave.Dim()
              << ")";
  Vector<BaseFloat> window;  // windowed waveform.
  double outer_min_lag = 1.0 / opts.max_f0 - (opts.upsample_filter_width /
                                              (2.0 * opts.resample_freq));
  double outer_max_lag = 1.0 / opts.min_f0 + (opts.upsample_filter_width /
                                              (2.0 * opts.resample_freq));

  int32 start = ceil(opts.resample_freq * outer_min_lag),
        end = floor(opts.resample_freq * outer_max_lag),
        num_initial_lags = end - start + 1;
  // num_initial_lags is the number of lags we initially compute-- evenly
  // spaced, integer lags-- before doing the resampling.

  Vector<BaseFloat> final_lags;
  SelectLags(opts, &final_lags);
  // final_lags is a list of the lags we want to resample the NCCF at;
  // these are log-spaced so we have finer resolution for smaller lags.

  int32 num_states = final_lags.Dim();
  Matrix<BaseFloat> nccf_pitch(num_frames, num_initial_lags),
      nccf_pov(num_frames, num_initial_lags);

  double sumsq = VecVec(processed_wave, processed_wave);
  double sum = processed_wave.Sum();
  double mean_square =
      sumsq / processed_wave.Dim() - pow(sum / processed_wave.Dim(), 2.0);

  double nccf_ballast_pitch =
      pow(mean_square * opts.NccfWindowSize(), 2) * opts.nccf_ballast;
  double nccf_ballast_pov = 0.0;

  Matrix<BaseFloat> windows(num_frames, opts.NccfWindowSize() + end);
  Matrix<BaseFloat> inner_prods(num_frames, num_initial_lags, kSetZero);
  Matrix<BaseFloat> norm_prods(num_frames, num_initial_lags, kSetZero);

  for (int32 r = 0; r < num_frames; r++) {  // r is frame index.
    ExtractFrame(processed_wave, r, opts, &window);
    SubVector<BaseFloat> windows_row(windows, r);
    windows_row.CopyFromVec(window);

    // compute nccf for pitch extraction
    Vector<BaseFloat> inner_prod(num_initial_lags), norm_prod(num_initial_lags);
    ComputeCorrelation(window, start, end, opts.NccfWindowSize(), &inner_prod,
                       &norm_prod);
    SubVector<BaseFloat> inner_prods_row(inner_prods, r);
    SubVector<BaseFloat> norm_prods_row(norm_prods, r);
    inner_prods_row.CopyFromVec(inner_prod);
    norm_prods_row.CopyFromVec(norm_prod);

    SubVector<BaseFloat> nccf_pitch_vec(nccf_pitch.Row(r));
    ProcessNccf(inner_prod, norm_prod, nccf_ballast_pitch, &(nccf_pitch_vec));
    // compute the Nccf for Probability of voicing estimation;
    // there is no ballast for this version of the NCCF.
    SubVector<BaseFloat> nccf_pov_vec(nccf_pov.Row(r));
    ProcessNccf(inner_prod, norm_prod, nccf_ballast_pov, &(nccf_pov_vec));
  }
  Vector<BaseFloat> final_lags_offset(final_lags);
  final_lags_offset.Add(-start / opts.resample_freq);

  // WriteKaldiObject<Matrix<BaseFloat> >(windows, "windows.mat", true);
  // WriteKaldiObject<Matrix<BaseFloat> >(inner_prods, "inner_prods.mat", true);
  // WriteKaldiObject<Matrix<BaseFloat> >(norm_prods, "norm_prods.mat", true);
  // WriteKaldiObject<Matrix<BaseFloat> >(nccf_pitch, "nccf_pitch.mat", true);
  // WriteKaldiObject<Matrix<BaseFloat> >(nccf_pov, "nccf_pov.mat", true);

  BaseFloat upsample_cutoff = opts.resample_freq * 0.5;
  ArbitraryResample resample(num_initial_lags, opts.resample_freq,
                             upsample_cutoff, final_lags_offset,
                             opts.upsample_filter_width);
  Matrix<BaseFloat> resampled_nccf_pitch(num_frames, num_states);
  resample.Resample(nccf_pitch, &resampled_nccf_pitch);
  Matrix<BaseFloat> resampled_nccf_pov(num_frames, num_states);
  resample.Resample(nccf_pov, &resampled_nccf_pov);

  resampled_nccf_pitch.Resize(num_frames - 2, num_states, kCopyData);
  resampled_nccf_pov.Resize(num_frames - 2, num_states, kCopyData);

  // WriteKaldiObject<Matrix<BaseFloat> >(resampled_nccf_pitch,
  //                                      "resampled_nccf_pitch.mat", true);
  // WriteKaldiObject<Matrix<BaseFloat> >(resampled_nccf_pov,
  //                                      "resampled_nccf_pov.mat", true);

  PitchExtractor pitch(opts, final_lags, num_states, num_frames - 2);
  pitch.FastViterbi(resampled_nccf_pitch);
  // pitch.FindBestPath will use the NCCF without the "ballast" term
  // when it notes the NCCF at each frame.
  pitch.FindBestPath(resampled_nccf_pov);

  // FIXME: remove copy
  Matrix<BaseFloat> output;
  output.Resize(num_frames - 2, 2);  // (pov, pitch)
  pitch.GetPitchAndPov(&output);

  cu_pitch_pov->Resize(output.NumRows(), output.NumCols());
  cu_pitch_pov->CopyFromMat(output);
}

ComputePitchCuda::ComputePitchCuda(const PitchExtractionOptions &opts)
    : opts(opts) {
  if (opts.max_frames_latency != 0 || opts.frames_per_chunk != 0 ||
      opts.simulate_first_pass_online != false || opts.recompute_frame != 500 ||
      opts.nccf_ballast_online != false) {
    KALDI_ERR << "ComputePitchCuda does not support online pitch extraction";
  }
}

void ProcessPitchCuda::ProcessPitch(const CuMatrix<BaseFloat> &cu_pitch_pov,
                                    CuMatrix<BaseFloat> *cu_features) {}

ProcessPitchCuda::ProcessPitchCuda(const ProcessPitchOptions &opts) {}

}  // namespace kaldi
