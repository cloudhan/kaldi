// online2/online-nnet2-feature-pipeline.cc

// Copyright    2013  Johns Hopkins University (author: Daniel Povey)

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

#include "cudadecoder/cuda-feature-pipeline.h"
#include "transform/cmvn.h"

namespace kaldi {

CudaFeatureAdapter::CudaFeatureAdapter(CudaMfcc* cuda_mfcc)
 : computed_(false)
 , sample_frequency_(-INFINITY)
 , mfcc_computer(cuda_mfcc)
{}

CudaFeatureAdapter::~CudaFeatureAdapter() {
  delete mfcc_computer;
}

int32 CudaFeatureAdapter::Dim() const {
  if(mfcc_computer) {
    return mfcc_computer->Dim();
  }

  KALDI_ERR << "Unknown cuda feature computer";
}

int32 CudaFeatureAdapter::NumFramesReady() const {
  return features_.NumRows();
}

bool CudaFeatureAdapter::IsLastFrame(int32 frame) const {
  return computed_ && frame == NumFramesReady() - 1;
}

BaseFloat CudaFeatureAdapter::FrameShiftInSeconds() const {
  KALDI_ERR << "NotImplemented";
}

void CudaFeatureAdapter::AcceptWaveform(BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &new_waveform) {
  if (new_waveform.Dim() == 0)
    return;  // Nothing to do.
  if (computed_)
    KALDI_ERR << "AcceptWaveform called after feature has been computed";
  if (sample_frequency_ < 0.0 || abs(sample_frequency_ - sampling_rate) < 0.00001 ) {
    sample_frequency_ = sampling_rate;
  }
  else {
    KALDI_ERR << "try to accept waveform with different sampling rate";
  }

  Vector<BaseFloat> appended_wave;

  appended_wave.Resize(waveform_.Dim() + new_waveform.Dim());
  if (waveform_.Dim() != 0)
    appended_wave.Range(0, waveform_.Dim())
        .CopyFromVec(waveform_);
  appended_wave.Range(waveform_.Dim(), new_waveform.Dim())
      .CopyFromVec(new_waveform);
  waveform_.Swap(&appended_wave);
}

void CudaFeatureAdapter::InputFinished() {
  if(computed_) {
    KALDI_ERR << "InputFinished called multiple times";
  }

  if(mfcc_computer) {
    CuVector<BaseFloat> tmp_cu_wave;
    tmp_cu_wave.Resize(waveform_.Dim());
    CuMatrix<BaseFloat> temp_cu_feature_out;
    tmp_cu_wave.CopyFromVec(waveform_);

    // NOTE: VTLN is not supported
    mfcc_computer->ComputeFeatures(tmp_cu_wave, sample_frequency_, 1.0, &temp_cu_feature_out);
    features_.Resize(temp_cu_feature_out.NumRows(), temp_cu_feature_out.NumCols());
    features_.CopyFromMat(temp_cu_feature_out);
  }
  else {
    KALDI_ERR << "Unknown cuda feature computer";
  }
  computed_ = true;
}

void CudaFeatureAdapter::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  KALDI_ASSERT(computed_);
  KALDI_ASSERT(frame < features_.NumRows());

  feat->CopyRowFromMat(features_, frame);
}

void CudaFeatureAdapter::GetFrames(const std::vector<int32> &frames,
                         MatrixBase<BaseFloat> *feats){
  KALDI_ASSERT(computed_);
  for(auto i : frames){
    KALDI_ASSERT(i < features_.NumRows());
  }
  KALDI_ASSERT(static_cast<int32>(frames.size()) == feats->NumRows());

  feats->CopyRows(features_, frames.data());
}


CudaFeaturePipeline::CudaFeaturePipeline(
    const OnlineNnet2FeaturePipelineInfo &info):
    info_(info) {
  if (info_.feature_type == "mfcc") {
    base_feature_ = new CudaFeatureAdapter(new CudaMfcc(info_.mfcc_opts));
  } else {
    KALDI_ERR << "Code error: invalid feature type " << info_.feature_type;
  }

  if (info_.add_pitch) {
    pitch_ = new OnlinePitchFeature(info_.pitch_opts);
    pitch_feature_ = new OnlineProcessPitch(info_.pitch_process_opts,
                                            pitch_);
    feature_plus_optional_pitch_ = new OnlineAppendFeature(base_feature_,
                                                           pitch_feature_);
  } else {
    pitch_ = NULL;
    pitch_feature_ = NULL;
    feature_plus_optional_pitch_ = base_feature_;
  }

  if (info_.use_ivectors) {
    ivector_feature_ = new OnlineIvectorFeature(info_.ivector_extractor_info,
                                                base_feature_);
    final_feature_ = new OnlineAppendFeature(feature_plus_optional_pitch_,
                                             ivector_feature_);
  } else {
    ivector_feature_ = NULL;
    final_feature_ = feature_plus_optional_pitch_;
  }
  dim_ = final_feature_->Dim();
}

int32 CudaFeaturePipeline::Dim() const { return dim_; }

bool CudaFeaturePipeline::IsLastFrame(int32 frame) const {
  return final_feature_->IsLastFrame(frame);
}

int32 CudaFeaturePipeline::NumFramesReady() const {
  return final_feature_->NumFramesReady();
}

void CudaFeaturePipeline::GetFrame(int32 frame,
                                          VectorBase<BaseFloat> *feat) {
  return final_feature_->GetFrame(frame, feat);
}

void CudaFeaturePipeline::UpdateFrameWeights(
    const std::vector<std::pair<int32, BaseFloat> > &delta_weights,
    int32 frame_offset) {
  if (frame_offset == 0) {
    IvectorFeature()->UpdateFrameWeights(delta_weights);
  } else {
    std::vector<std::pair<int32, BaseFloat> > offset_delta_weights;
    for (size_t i = 0; i < delta_weights.size(); i++) {
      offset_delta_weights.push_back(std::make_pair(
          delta_weights[i].first + frame_offset, delta_weights[i].second));
    }
    IvectorFeature()->UpdateFrameWeights(offset_delta_weights);
  }
}

void CudaFeaturePipeline::SetAdaptationState(
    const OnlineIvectorExtractorAdaptationState &adaptation_state) {
  if (info_.use_ivectors) {
    ivector_feature_->SetAdaptationState(adaptation_state);
  }
  // else silently do nothing, as there is nothing to do.
}

void CudaFeaturePipeline::GetAdaptationState(
    OnlineIvectorExtractorAdaptationState *adaptation_state) const {
  if (info_.use_ivectors) {
    ivector_feature_->GetAdaptationState(adaptation_state);
  }
  // else silently do nothing, as there is nothing to do.
}


CudaFeaturePipeline::~CudaFeaturePipeline() {
  // Note: the delete command only deletes pointers that are non-NULL.  Not all
  // of the pointers below will be non-NULL.
  // Some of the online-feature pointers are just copies of other pointers,
  // and we do have to avoid deleting them in those cases.
  if (final_feature_ != feature_plus_optional_pitch_)
    delete final_feature_;
  delete ivector_feature_;
  if (feature_plus_optional_pitch_ != base_feature_)
    delete feature_plus_optional_pitch_;
  delete pitch_feature_;
  delete pitch_;
  delete base_feature_;
}

void CudaFeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
  if (pitch_)
    pitch_->AcceptWaveform(sampling_rate, waveform);
}

void CudaFeaturePipeline::InputFinished() {
  base_feature_->InputFinished();
  if (pitch_)
    pitch_->InputFinished();
}


}  // namespace kaldi
