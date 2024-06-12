import numpy as np
import h5py
import os
from shutil import rmtree
from tqdm import tqdm
import soundfile as sf
import numpy as np
from pqdm.processes import pqdm as pqdm_p
from pqdm.threads import pqdm as pqdm_t
from scipy.signal import convolve

from util import *

import argparse

def store_audio(rir_idx, voc_idx, path, db):
    audio, loc, voc_idx = convolve_audio((rir_idx, voc_idx), db)
    num_mics = len(audio)
    with h5py.File(path, "w") as f:
        audio = f.create_dataset("audio", shape=(0,num_mics), maxshape=(None,num_mics))
        length_idx = f.create_dataset("length_idx", shape=(1,), data=np.array([0]), maxshape=(None,))
        locations = f.create_dataset("locations", shape=(0,2), maxshape=(None,2))
        stimulus_identities = f.create_dataset("stimulus_identities", shape=(0,), maxshape=(None,))
        if stimulus_names is not None:
            f.create_dataset("stimulus_names", shape=stimulus_names.shape, data=stimulus_names)
            
        aud_len = audio.shape[1]
        audio.resize(audio.shape[0] + aud_len, axis=0)
        audio[-aud_len:] = audio.T

        length_idx.resize(length_idx.shape[0]+1, axis=0)
        length_idx[-1:] = length_idx[-2:-1][0] + aud_len

        locations.resize(locations.shape[0]+1, axis=0)
        locations[-1:] = loc[None,:]

        stimulus_identities.resize(stimulus_identities.shape[0]+1, axis=0)
        stimulus_identities[-1:] = voc_idx
    return path

#define function for convolving rir w/ vocalization
def convolve_audio(idx_pair, db):
    rir_idx, voc_idx = idx_pair
    rir, loc = rir_loc_for_index(rir_idx, db)
    voc = vocalizations[voc_idx][None, :]
    audio = convolve(rir, voc, mode="full")
    return audio, loc, voc_idx

def merge_aud_dbs(save_path, piece_paths):
    with h5py.File(piece_paths[0], 'r') as f:
        num_mics = f['audio'].shape[1]
        
    with h5py.File(save_path, 'w') as f:
        audio = f.create_dataset("audio", shape=(0,num_mics), maxshape=(None,num_mics))
        idx = f.create_dataset("length_idx", shape=(1,), data=np.array([0]), maxshape=(None,))
        loc = f.create_dataset("locations", shape=(0,2), maxshape=(None,2))
        stimulus_identities = f.create_dataset("stimulus_identities", shape=(0,), maxshape=(None,))
        if stimulus_names is not None:
            f.create_dataset("stimulus_names", shape=stimulus_names.shape, data=stimulus_names)

        for path in tqdm(piece_paths):
            with h5py.File(path, 'r') as g:
                aud_len = g['audio'].shape[0]
                audio.resize(audio.shape[0]+aud_len, axis=0)
                audio[-aud_len:] = g['audio'][:]

                n_aud = g['locations'].shape[0]
                loc.resize(loc.shape[0]+n_aud, axis=0)
                loc[-n_aud:] = g['locations'][:]

                idx_end = idx[-1]
                idx.resize(idx.shape[0]+n_aud, axis=0)
                idx[-n_aud:] = g['length_idx'][1:] + idx_end
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "save_path",
        type=str,
        help="path to save audio dataset",
    )

    parser.add_argument(
        dest="rir_path",
        type=str,
        help="path to load rir dataset",
        required=True,
    )
    #/vast/ci411/gerbil_data/rir_datasets/default.h5

    parser.add_argument(
        dest="voc_path",
        type=str,
        help="path to load vocalizations",
        required=True,
    )
    #/vast/ci411/gerbil_data/vocal_datasets/raw/speaker_resampled
    
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        help="path to save rir dataset parts",
        required=False,
        default=None,
    )
    
    parser.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        help="number of jobs to use in parallized process",
        required=False,
        default=16,
    )

    parser.add_argument(
        "--all-vocs",
        dest="all_vocs",
        type=bool,
        help="use all vocalizations per rir (if not, choose one random per rir)",
        required=False,
        default=False,
    )

    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        help="limit of RIRs to use",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--use-threads",
        dest="use_threads",
        type=bool,
        help="use threads for parallelization (otherwise, use processes)",
        required=False,
        default=False,
    )


    args = parser.parse_args()
    
    
    if args.save_dir is None:
        args.save_dir = args.save_path.split('.')[0]
        print(f"Storing parts at {args.save_dir}")
    
    if os.path.exists(args.save_dir):
        rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    
    if args.use_threads:
        pqdm = pqdm_t
    else:
        pqdm = pqdm_p


    #load vocalizations (normalized)
    print("Loading vocalizations...")
    vocalizations = []
    for file in os.listdir(args.voc_path):
        aud, sr = sf.read(os.path.join(args.voc_path, file))
        aud = (aud - np.average(aud)) / np.std(aud)
        vocalizations.append(aud)
    
    #load RIR dataset
    n_rirs = get_n_rirs(args.rir_path)
    
    #either convolve with each vocalization or one per rir
    with h5py.File(args.rir_path, 'r') as f:
        if args.all_vocs:
            #doesn't work
            print("Not functional")
        else:
            p_args = [[i,np.random.randint(0, len(vocalizations)),
                      os.path.join(args.save_dir,f'part{i}.part'),
                      f] for i in range(n_rirs)]

        #debugging limit    
        if args.limit is not None:
            p_args = pairs[:args.limit]
        #convolve rirs/vocalizations
        print("Testing...")
        _ = store_audio(*p_args[0])
        print("Storing...")
        paths = pqdm(p_args, store_audio, n_jobs=args.n_jobs, argument_type='args')
        #gather results to appropriate format for .h5 storage
    print(f"Storing results at {args.save_path}")
    stimulus_names = np.array([item.split('.')[0] for item in os.listdir(args.voc_path)], dtype=object)
    merge_aud_dbs(args.save_path, paths)
    rmtree(args.save_dir)
    