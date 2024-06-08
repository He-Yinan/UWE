# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np
import time

assert sizeof(int) == sizeof(np.int32_t)
assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "feat_dist_gpu.hpp":
    void _feat_dist_gpu(np.float32_t* dist_host, np.float32_t* feat_host, np.float32_t* sel_feat, int feat_num, int feat_dim, int metric, int device_id)

def feat_dist_gpu(np.ndarray[np.float32_t, ndim=2] all_feats, np.ndarray[np.float32_t, ndim=1] sel_feat, np.int32_t metric, np.int32_t device_id=0):
    
    cdef int feat_num = all_feats.shape[0]
    cdef int feat_dim = all_feats.shape[1]
   
    cdef np.ndarray[np.float32_t, ndim=1] \
        dist = np.zeros(feat_num, dtype=np.float32)

    _feat_dist_gpu(&dist[0], &all_feats[0, 0],  &sel_feat[0], feat_num, feat_dim, metric, device_id)
   
    return dist
