import numpy as np
cimport numpy as np

def maximum_path(np.ndarray[np.float32_t, ndim=2] value,
                 np.ndarray[np.int32_t, ndim=1] t_x,
                 np.ndarray[np.int32_t, ndim=1] t_y):
    cdef int b = value.shape[0]
    cdef int x_max = value.shape[1]
    cdef int y_max = value.shape[2] if value.ndim > 2 else 1
    return value
