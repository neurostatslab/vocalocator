import h5py

def get_n_rirs(path):
    with h5py.File(path, 'r') as f:
        n_rirs = f['locations'].shape[0]
    return n_rirs

#define helper function for extracting RIR/location
def rir_loc_for_index(idx, db):
    start, end = db['rir_length_idx'][idx : idx + 2]
    rir = db['rir'][start:end, :]
    loc = db['locations'][idx]
    return rir.T, loc