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

/// @file
/// This file contains a different version of the feature-extraction pipeline in
/// \ref online-feature-pipeline.h, specialized for use in neural network
/// decoding with iVectors.  Our recipe is that we extract iVectors that will
/// be used as an additional input to the neural network, in addition to
/// a window of several frames of spliced raw features (MFCC, PLP or filterbanks).
/// The iVectors are extracted on top of a (splice+LDA+MLLT) feature pipeline,
/// with the added complication that the GMM posteriors used for the iVector
/// extraction are obtained with a version of the features that has online
/// cepstral mean (and optionally variance) normalization, whereas the stats for
/// iVector are accumulated with a non-mean-normalized version of the features.
/// The idea here is that we want the iVector to learn the mean offset, but
/// we want the posteriors to be somewhat invariant to mean offsets.
///
/// Most of the logic for the actual iVector estimation is in \ref
/// online-ivector-feature.h, this header contains mostly glue.
///
/// Although the name of this header mentions nnet2, actually the code is
/// used in the online decoding with nnet3 also.


/// This configuration class is to set up CudaFeaturePipelineInfo, which
/// in turn is the configuration class for CudaFeaturePipeline.
/// Instead of taking the options for the parts of the feature pipeline
/// directly, it reads in the names of configuration classes.
// struct CudaFeaturePipelineConfig {
//   std::string feature_type;  // "plp" or "mfcc" or "fbank"
//   std::string mfcc_config;
//   std::string plp_config;
//   std::string fbank_config;

//   // Note: if we do add pitch, it will not be added to the features we give to
//   // the iVector extractor but only to the features we give to the neural
//   // network, after the base features but before the iVector.  We don't think
//   // the iVector will be particularly helpful in normalizing the pitch features,
//   // and we wanted to avoid complications with things like online CMVN.
//   bool add_pitch;

//   // the following contains the type of options that you could give to
//   // compute-and-process-kaldi-pitch-feats.
//   std::string online_pitch_config;

//   // The configuration variables in ivector_extraction_config relate to the
//   // iVector extractor and options related to it, see type
//   // OnlineIvectorExtractionConfig.
//   std::string ivector_extraction_config;

//   // Config that relates to how we weight silence for (ivector) adaptation
//   // this is registered directly to the command line as you might want to
//   // play with it in test time.
//   OnlineSilenceWeightingConfig silence_weighting_config;

//   CudaFeaturePipelineConfig():
//       feature_type("mfcc"), add_pitch(false) { }


//   void Register(OptionsItf *opts) {
//     opts->Register("feature-type", &feature_type,
//                    "Base feature type [mfcc, plp, fbank]");
//     opts->Register("mfcc-config", &mfcc_config, "Configuration file for "
//                    "MFCC features (e.g. conf/mfcc.conf)");
//     opts->Register("plp-config", &plp_config, "Configuration file for "
//                    "PLP features (e.g. conf/plp.conf)");
//     opts->Register("fbank-config", &fbank_config, "Configuration file for "
//                    "filterbank features (e.g. conf/fbank.conf)");
//     opts->Register("add-pitch", &add_pitch, "Append pitch features to raw "
//                    "MFCC/PLP/filterbank features [but not for iVector extraction]");
//     opts->Register("online-pitch-config", &online_pitch_config, "Configuration "
//                    "file for online pitch features, if --add-pitch=true (e.g. "
//                    "conf/online_pitch.conf)");
//     opts->Register("ivector-extraction-config", &ivector_extraction_config,
//                    "Configuration file for online iVector extraction, "
//                    "see class OnlineIvectorExtractionConfig in the code");
//     silence_weighting_config.RegisterWithPrefix("ivector-silence-weighting", opts);
//   }
// };


/// This class is responsible for storing configuration variables, objects and
/// options for CudaFeaturePipeline (including the actual LDA and
/// CMVN-stats matrices, and the iVector extractor, which is a member of
/// ivector_extractor_info.  This class does not register options on the command
/// line; instead, it is initialized from class CudaFeaturePipelineConfig
/// which reads the options from the command line.  The reason for structuring
/// it this way is to make it easier to configure from code as well as from the
/// command line, as well as for easiter multithreaded operation.
// struct CudaFeaturePipelineInfo {
//   CudaFeaturePipelineInfo():
//       feature_type("mfcc"), add_pitch(false) { }

//   CudaFeaturePipelineInfo(
//       const CudaFeaturePipelineConfig &config);

//   BaseFloat FrameShiftInSeconds() const;

//   std::string feature_type;  // "mfcc" or "plp" or "fbank"

//   MfccOptions mfcc_opts;  // options for MFCC computation,
//                           // if feature_type == "mfcc"
//   PlpOptions plp_opts;  // Options for PLP computation, if feature_type == "plp"
//   FbankOptions fbank_opts;  // Options for filterbank computation, if
//                             // feature_type == "fbank"

//   bool add_pitch;
//   PitchExtractionOptions pitch_opts;  // Options for pitch extraction, if done.
//   ProcessPitchOptions pitch_process_opts;  // Options for pitch post-processing


//   // If the user specified --ivector-extraction-config, we assume we're using
//   // iVectors as an extra input to the neural net.  Actually, we don't
//   // anticipate running this setup without iVectors.
//   bool use_ivectors;
//   OnlineIvectorExtractionInfo ivector_extractor_info;

//   // Config for weighting silence in iVector adaptation.
//   // We declare this outside of ivector_extractor_info... it was
//   // just easier to set up the code that way; and also we think
//   // it's the kind of thing you might want to play with directly
//   // on the command line instead of inside sub-config-files.
//   OnlineSilenceWeightingConfig silence_weighting_config;

//   int32 IvectorDim() { return ivector_extractor_info.extractor.IvectorDim(); }
//  private:
//   KALDI_DISALLOW_COPY_AND_ASSIGN(CudaFeaturePipelineInfo);
// };


class CudaFeatureAdapter: public OnlineFeatureInterface {
 public:
  explicit CudaFeatureAdapter(CudaMfcc* cuda_mfcc);

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
