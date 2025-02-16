// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ONNXRUNTIME_REGISTER_H
#define ONNXRUNTIME_REGISTER_H
#include <onnxruntime_c_api.h>

// #include "core/macro.h"

#ifdef __cplusplus
extern "C" {
#endif

// MMDEPLOY_API OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
//                                                        const OrtApiBase *api);
 OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                       const OrtApiBase *api);

#ifdef __cplusplus
}
#endif
#endif  // ONNXRUNTIME_REGISTER_H
