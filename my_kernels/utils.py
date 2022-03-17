

def blocks_per_grid(data_size, tpb):
    return (data_size + (tpb - 1)) // tpb

def auto_build(f, data_size, tpb):
    bpg = blocks_per_grid(data_size, tpb)
    return f[bpg, tpb]