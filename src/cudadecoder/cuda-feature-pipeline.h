// online2/online-nnet2-feature-pipeline.h

// Copyright 2013-2014   Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_CUDA_FEATURE_PIPELINE_H_
#define KALDI_CUDA_FEATURE_PIPELINE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/online-feature.h"
#include "feat/pitch-functions.h"
#include "online2/online-ivector-feature.h"
#include "cudafeat/feature-mfcc-cuda.h"
#include "online2/online-nnet2-feature-pipeline.h"

namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{

class CudaFeatureAdapter: public OnlineFeatureInterface {
 public:
  explicit CudaFeatureAdapter(CudaMfcc* cuda_mfcc);
  virtual ~CudaFeatureAdapter();

  virtual int32 Dim() const;
  virtual int32 NumFramesReady() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual BaseFloat FrameShiftInSeconds() const;


  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const VectorBase<BaseFloat> &waveform);
  virtual void InputFinished();

  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  virtual void GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats);

private:
  bool computed_;
  BaseFloat sample_frequency_;
  Vector<BaseFloat> waveform_;
  Matrix<BaseFloat> features_;
  CudaMfcc* mfcc_computer;
};

/// CudaFeaturePipeline is a class that's responsible for putting
/// together the various parts of the feature-processing pipeline for neural
/// networks, in an online setting.  The recipe here does not include fMLLR;
/// instead, it assumes we're giving raw features such as MFCC or PLP or
/// filterbank (with no CMVN) to the neural network, and optionally augmenting
/// these with an iVector that describes the speaker characteristics.  The
/// iVector is extracted using class OnlineIvectorFeature (see that class for
/// more info on how it's done).
/// No splicing is currently done in this code, as we're currently only supporting
/// the nnet2 neural network in which the splicing is done inside the network.
/// Probably our strategy for nnet1 network conversion would be to convert to nnet2
/// and just add layers to do the splicing.
class CudaFeaturePipeline: public OnlineFeatureInterface {
 public:
  /// Constructor from the "info" object.  After calling this for a
  /// non-initial utterance of a speaker, you may want to call
  /// SetAdaptationState().
  explicit CudaFeaturePipeline(
      const OnlineNnet2FeaturePipelineInfo &info);

  /// Member functions from OnlineFeatureInterface:

  /// Dim() will return the base-feature dimension (e.g. 13 for normal MFCC);
  /// plus the pitch-feature dimension (e.g. 3), if used; plus the iVector
  /// dimension, if used.  Any frame-splicing happens inside the neural-network
  /// code.
  virtual int32 Dim() const;

  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);

  /// If you are downweighting silence, you can call
  /// OnlineSilenceWeighting::GetDeltaWeights and supply the output to this
  /// class using UpdateFrameWeights().  The reason why this call happens
  /// outside this class, rather than this class pulling in the data weights,
  /// relates to multi-threaded operation and also from not wanting this class
  /// to have excessive dependencies.
  ///
  /// You must either always call this as soon as new data becomes available,
  /// ideally just after calling AcceptWaveform(), or never call it for the
  /// lifetime of this object.
  void UpdateFrameWeights(
      const std::vector<std::pair<int32, BaseFloat> > &delta_weights,
      int32 frame_offset = 0);

  /// Set the adaptation state to a particular value, e.g. reflecting previous
  /// utterances of the same speaker; this will generally be called after
  /// Copy().
  void SetAdaptationState(
      const OnlineIvectorExtractorAdaptationState &adaptation_state);


  /// Get the adaptation state; you may want to call this before destroying this
  /// object, to get adaptation state that can be used to improve decoding of
  /// later utterances of this speaker.  You might not want to do this, though,
  /// if you have reason to believe that something went wrong in the recognition
  /// (e.g., low confidence).
  void GetAdaptationState(
      OnlineIvectorExtractorAdaptationState *adaptation_state) const;


  /// Accept more data to process.  It won't actually process it until you call
  /// GetFrame() [probably indirectly via (decoder).AdvanceDecoding()], when you
  /// call this function it will just copy it).  sampling_rate is necessary just
  /// to assert it equals what's in the config.
  void AcceptWaveform(BaseFloat sampling_rate,
                      const VectorBase<BaseFloat> &waveform);

  BaseFloat FrameShiftInSeconds() const { return info_.FrameShiftInSeconds(); }

  /// If you call InputFinished(), it tells the class you won't be providing any
  /// more waveform.  This will help flush out the last few frames of delta or
  /// LDA features, and finalize the pitch features (making them more
  /// accurate)... although since in neural-net decoding we don't anticipate
  /// rescoring the lattices, this may not be much of an issue.
  void InputFinished();

  // This function returns the iVector-extracting part of the feature pipeline
  // (or NULL if iVectors are not being used); the pointer ownership is retained
  // by this object and not transferred to the caller.  This function is used in
  // nnet3, and also in the silence-weighting code used to exclude silence from
  // the iVector estimation.
  OnlineIvectorFeature *IvectorFeature() {
    return ivector_feature_;
  }

  // A const accessor for the iVector extractor. Returns NULL if iVectors are
  // not being used.
  const OnlineIvectorFeature *IvectorFeature() const {
    return ivector_feature_;
  }

  // This function returns the part of the feature pipeline that would be given
  // as the primary (non-iVector) input to the neural network in nnet3
  // applications.
  OnlineFeatureInterface *InputFeature() {
    return feature_plus_optional_pitch_;
  }

  virtual ~CudaFeaturePipeline();
 private:

  const OnlineNnet2FeaturePipelineInfo &info_;

  CudaFeatureAdapter *base_feature_;        // MFCC/PLP/filterbank

  OnlinePitchFeature *pitch_;              // Raw pitch, if used
  OnlineProcessPitch *pitch_feature_;  // Processed pitch, if pitch used.


  // feature_plus_pitch_ is the base_feature_ appended (OnlineAppendFeature)
  /// with pitch_feature_, if used; otherwise, points to the same address as
  /// base_feature_.
  OnlineFeatureInterface *feature_plus_optional_pitch_;

  OnlineIvectorFeature *ivector_feature_;  // iVector feature, if used.

  // final_feature_ is feature_plus_optional_pitch_ appended
  // (OnlineAppendFeature) with ivector_feature_, if ivector_feature_ is used;
  // otherwise, points to the same address as feature_plus_optional_pitch_.
  OnlineFeatureInterface *final_feature_;

  // we cache the feature dimension, to save time when calling Dim().
  int32 dim_;
};

/// @} End of "addtogroup cudadecoder"
}  // namespace kaldi

#endif  // KALDI_CUDA_FEATURE_PIPELINE_H_
