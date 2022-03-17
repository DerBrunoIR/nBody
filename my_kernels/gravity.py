from numba import cuda, types


MINMAL_DIST = 1 #10 ** -9
MAX_DIM = 3

# Kernels

# Best tpb: 320
@cuda.jit
def calculate_accelerations_with_mass(global_positions, global_masses, global_results):
    """
    gloabl_positions: np.ndarray, position-vectors, shape=(n, dim)
    global_results: np.ndarray, velocitie-vectors, shape=(n, dim)
    global_masses: np.ndarray, masses-vectors, shape=(n, )
    """
    assert global_positions.shape[0] == global_masses.shape[0] == global_results.shape[0], ValueError("All argument arrays musst have the same length!")
    dtype = types.float64

    # thread unic values
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    idx = tx + bx * cuda.blockDim.x
    
    # references
    dim = global_positions.shape[1]
    n = global_positions.shape[0]


    # stop threas
    if idx >= n or idx >= len(global_masses) or idx >= len(global_results):
        return


    # shared arrays
    shared_pos= cuda.shared.array(MAX_DIM, dtype)
    shared_mass = cuda.shared.array(1, dtype)

    # local arrays
    local_total_acc = cuda.local.array(MAX_DIM, dtype)
    local_pos = cuda.local.array(MAX_DIM, dtype)
    local_dif = cuda.local.array(MAX_DIM, dtype)

    # set default values
    for i in range(dim):
        local_total_acc[i] = 0
        local_pos[i] = global_positions[idx][i]
        local_dif[i] = 0

    # iterate over all global_positions 
    for n in range(n):
        
        # load next pos, mass into shared memory.
        cuda.syncthreads()
        # load next pos
        for i in range(dim):
            shared_pos[i] = global_positions[n][i]
        
        # load next mass
        shared_mass[0] = global_masses[n]
        cuda.syncthreads()
        
        if n == idx:
            continue

        # calculate differenc vector between local_pos and shared_pos
        for i in range(dim):
            local_dif[i] = local_pos[i] - shared_pos[i]
        
        # calculate distance
        squared_dist = MINMAL_DIST
        for i in range(dim):
            squared_dist += local_dif[i]**2
        
        # calculate acceleration
        acc = - 1 * shared_mass[0] / squared_dist ** (3/2)
        # acc = - 1 * shared_mass[0] / squared_dist

        # calculate acc vector
        for i in range(dim):
            local_total_acc[i] += acc * local_dif[i]

    # write result into the global result array
    for i in range(dim):
        global_results[idx][i] += local_total_acc[i]
        
