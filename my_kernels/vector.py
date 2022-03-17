from numba import cuda, types


# best tpb: 640
@cuda.jit
def add_vector_arrays(arr1, arr2, s):
    """
    v1[i] += v2[i] * s
    """
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    idx = tx + bx * cuda.blockDim.x
    dim = arr1.shape[1]
    
    if idx >= arr1.shape[0]:
        return

    for i in range(dim):
        arr1[idx][i] += arr2[idx][i] * s