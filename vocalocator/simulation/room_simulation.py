import numpy as np
import yaml
import pyroomacoustics as pra
import sys
import h5py
from tqdm import tqdm
from pqdm.processes import pqdm
import os
from shutil import rmtree

import argparse

from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

cardioid_map = {'CARDIOID' : DirectivityPattern.CARDIOID,
               'FIGURE_EIGHT' : DirectivityPattern.FIGURE_EIGHT,
               'HYPERCARDIOID' : DirectivityPattern.HYPERCARDIOID,
               'OMNI' : DirectivityPattern.OMNI,
               'SUBCARDIOID' : DirectivityPattern.SUBCARDIOID}

default_mic_pos = np.array(
        [
            [0.02064812, 0.05017473, 0.28968258],
            [0.55085188, 0.05017473, 0.28968258],
            [0.55085188, 0.30542527, 0.28968258],
            [0.02064812, 0.30542527, 0.28968258]
        ])

default_mic_dir = np.array(
        [
            [ 0.37556186,  0.7675516,  -0.51943994],
            [-0.36181653,  0.77332777, -0.52062756],
            [-0.36777207, -0.7940425,  -0.4839836 ],
            [ 0.35280470, -0.779317,   -0.5178743 ]
        ])

#room absorption bugfix via aramis
unscaled_coeffs = [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 100.0, 530.0, 800.0]
scaled_coeffs = [c * 1e-3 for c in unscaled_coeffs]
center_freqs=[
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        32000.0,
        64000.0,
    ]

def arena_random_point(arena_dims,z_offset=5e-2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x = rng.uniform(1e-2, arena_dims[0] - 1e-2)
    y = rng.uniform(1e-2, arena_dims[1] - 1e-2)
    # z = rng.uniform(1e-2, 5e-2)
    z = 5e-2
    return np.array([x, y, z])

def pad_rir(rir):
    # assuming only one sound source
    flat_rirs = [r[0] for r in rir]
    max_length = max([len(r) for r in flat_rirs])
    padded_rirs = [np.pad(r, (0, max_length - len(r))) for r in flat_rirs]
    return np.stack(padded_rirs, axis=0)

def get_scaled_pos(config, pos):
    #generate scaled position
    r_width, r_length, _ = config['room']['dim']
    scaled_sound_source_pos = (
                pos - np.array([r_width / 2, r_length / 2, 0])
            ) * 1000
    return scaled_sound_source_pos[:2]

def construct_room(config):
    #pull room dimensions from config
    room_config = config['room']
    room_jitter = room_config['jitter']
    
    #load room geo
    room_dims = np.array(room_config['dim'])
    r_offset = room_config['ceil_offset']
    
    #vary dims if jitter enabled
    if room_jitter['dims'] is not None:
        room_dims += np.random.normal(scale=room_jitter['dims'],size=3)
    
    #load absorption coeffs for surfaces
    wall_abs = room_config['wall_abs']
    flor_abs = room_config['flor_abs']
    ceil_abs = room_config['ceil_abs']
    
    #vary absorption if jitter enabled
    if room_jitter['abs'] is not None:
        wall_abs += np.random.normal(scale=room_jitter['abs'])
        flor_abs += np.random.normal(scale=room_jitter['abs'])
        ceil_abs += np.random.normal(scale=room_jitter['abs'])
        
        wall_abs = np.max([wall_abs, 1])
        flor_abs = np.max([flor_abs, 1])
        ceil_abs = np.max([ceil_abs, 1])       

    absorption_arr = [wall_abs] * 4 + [
                        flor_abs,
                        ceil_abs,
                     ]
    

    scattering = room_config['scattering']
    if room_jitter['scattering'] is not None:
        scattering += np.random.normal(scale=room_jitter['scattering'])
    
    air_abs = room_config['air_abs']
    
    sr = room_config['sr']
    max_order = room_config['max_order']
    
    r_width, r_length, r_height = room_dims
    
    #compute floor corners
    floor_corners = np.array(
        [
            [0, 0, 0],
            [r_width, 0, 0],
            [r_width, r_length, 0],
            [0, r_length, 0],
        ]
    )
    
    #compute ceil corners w/ offset
    ceiling_corners = np.array(
        [
            [0 - r_offset, 0 - r_offset, r_height],
            [r_width + r_offset, 0 - r_offset, r_height],
            [r_width + r_offset, r_length + r_offset, r_height],
            [0 - r_offset, r_length + r_offset, r_height],
        ]
    )
    #join all vertices
    all_vertices = np.concatenate((floor_corners, ceiling_corners), axis=0)
    
    #index walls by above vertice array
    wall_vertex_indices = np.array(
        [
            [0, 4, 7, 3],  # left wall
            [1, 2, 6, 5],  # right wall
            [0, 1, 5, 4],  # bottom wall
            [3, 7, 6, 2],  # top wall
            [0, 1, 2, 3],  # floor
            [4, 5, 6, 7],  # ceiling
        ]
    )
    
    #compute materials via absorption/scattering coefficients
    materials = [pra.Material(a, scattering) for a in absorption_arr]
    
    #build walls
    walls = [
        pra.room.wall_factory(
            corners=all_vertices[v_idx, :].T,
            absorption=m.absorption_coeffs,
            scattering=m.scattering_coeffs,
            name="wall_{n}".format(n=n),
        )
        for n, (v_idx, m) in enumerate(zip(wall_vertex_indices, materials))
    ]
    # Construct room
    room = pra.room.Room(
        walls=walls,
        fs=sr,
        max_order=max_order,
        use_rand_ism=False,
        ray_tracing=False,
        air_absorption=True,
    )
    
    #modify air absorption, if included
    if air_abs is not None:
        coeffs, center_freqs = air_abs
        room.simulator_state["air_abs_needed"] = True
        absorption_bands = room.octave_bands(coeffs=coeffs,
                                             center_freqs=center_freqs)
        room.air_absorption = absorption_bands
    
    #add microphones
    mic_config = config['mics']
    microphone_pos = np.array(mic_config['mic_pos'])

    if mic_config['mic_dir'] is not None:
        mic_direction_vectors = mic_config['mic_dir']
    else:
        room_center_3d = np.array(
            [
                r_width / 2,
                r_length / 2,
                0,
            ]
        )

        # Point microphones at room_center_3d
        mic_direction_vectors = room_center_3d - microphone_pos
    mic_direction_vectors /= np.linalg.norm(
        mic_direction_vectors, axis=1, keepdims=True
    )
    
    #compute directivities
    all_directivities = []
    pattern = cardioid_map[mic_config['mic_pattern']]
    for i in range(mic_direction_vectors.shape[0]):
        orientation = pra.directivities.DirectionVector(
            azimuth=np.arctan2(
                mic_direction_vectors[i, 1], mic_direction_vectors[i, 0]
            ),
            colatitude=np.arccos(mic_direction_vectors[i, 2]),
            degrees=False,
        )
        directivity = pra.directivities.CardioidFamily(orientation, pattern)
        all_directivities.append(directivity)
    
    mic_array = pra.MicrophoneArray(
        microphone_pos.T, fs=sr, directivity=all_directivities
    )
    room.add_microphone_array(mic_array)
    
    return room

def compute_rirs(config, n_rir):
    rir_db = []
    src_config = config['srcs']
    
    i = 0
    while i<n_rir:
        room = construct_room(config)
        pos = arena_random_point(config['room']['dim'],z_offset=src_config['z_offset'])
        
        azi = 360 * np.random.rand()
        col = 180 * np.random.rand()    
        dir_obj = CardioidFamily(orientation=DirectionVector(azimuth=azi, colatitude=col, degrees=True),
                                 pattern_enum=cardioid_map[src_config['src_pattern']])
    
        while True:
            try:
                room.add_source(pos)
                break
            except:
                #SHOULDN'T BE AN ISSUE assuming user picks points inside their room
                pos = arena_random_point(config['room']['dim'],z_offset=src_config['z_offset'])
                continue
            
        room.compute_rir()
        rir = pad_rir(room.rir)
        if (rir>0).any():
            rir_db.append((rir, get_scaled_pos(config, pos)))
            i += 1
    return rir_db

def store_rirs(path, rir_db):
    num_mics = rir_db[0][0].shape[0]
    full_rirs = np.ascontiguousarray(
        np.concatenate([r[0] for r in rir_db], axis=1).T
    )
    rir_lengths = np.array([r[0].shape[1] for r in rir_db])
    rir_full_length = rir_lengths.sum()
    with h5py.File(path, "w") as f:
        f.create_dataset("rir", shape=(rir_full_length, num_mics), data=full_rirs)
        f.create_dataset(
            "locations",
            shape=(len(rir_db), 2),
            data=np.stack([r[1] for r in rir_db], axis=0),
        )
        f.create_dataset("rir_length_idx", data=np.cumsum(np.insert(rir_lengths, 0, 0)))

def store_rir_part(config, n_rirs, path):
    rir_db = compute_rirs(config, n_rirs)
    store_rirs(path, rir_db)
    return path

def merge_rir_dbs(save_path, piece_paths):
    
    with h5py.File(piece_paths[0], 'r') as f:
        num_mics = f['rir'].shape[1]
        
    with h5py.File(save_path, 'w') as f:
        rir = f.create_dataset("rir",
                         shape=(0, num_mics),
                         maxshape=(None,num_mics))
        loc = f.create_dataset("locations",
                         shape=(0, 2),
                         maxshape=(None,2))
        idx = f.create_dataset("rir_length_idx",
                         shape=(1,),
                         maxshape=(None,),
                         data=np.array([0]))

        for path in piece_paths:
            with h5py.File(path, 'r') as g:
                rir_len = g['rir'].shape[0]
                rir.resize(rir.shape[0]+rir_len, axis=0)
                rir[-rir_len:] = g['rir'][:]

                n_rirs = g['locations'].shape[0]
                loc.resize(loc.shape[0]+n_rirs, axis=0)
                loc[-n_rirs:] = g['locations'][:]

                idx_end = idx[-1]
                idx.resize(idx.shape[0]+n_rirs, axis=0)
                idx[-n_rirs:] = g['rir_length_idx'][1:] + idx_end

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        dest="save_path",
        type=str,
        help="path to save rir dataset",
        required=True
    )
    
    parser.add_argument(
        "--load-yaml",
        dest="load_yaml",
        type=str,
        help="path to load yaml conf",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--save-yaml",
        dest="save_yaml",
        type=str,
        help="path to save yaml conf",
        required=False,
        default=None,
    )
    
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        help="path to save rir dataset parts",
        default=None,
    )

    parser.add_argument(
        "--compute-rirs",
        dest="compute_rirs",
        type=bool,
        help="option to compute rirs (troubleshoooting)",
        required=False,
        default=True,
    )

    parser.add_argument(
        "--room-dims",
        dest="room_dims",
        type=list,
        help="sets room dimensions",
        required=False,
        default=[0.5715, 0.3556, 0.3683],
    )

    parser.add_argument(
        "--ceil-offset",
        dest="ceil_offset",
        type=float,
        help="sets ceiling offest (compared to base)",
        required=False,
        default=0.0254,
    )

    parser.add_argument(
        "--wall-abs",
        dest="wall_abs",
        type=float,
        help="sets wall absorption coefficient",
        required=False,
        default=0.14746730029582977,
    )

    parser.add_argument(
        "--flor-abs",
        dest="flor_abs",
        type=float,
        help="sets floor absorption coefficient",
        required=False,
        default=0.6360408663749695,
    )

    parser.add_argument(
        "--ceil-abs",
        dest="ceil_abs",
        type=float,
        help="sets ceiling absorption coefficient",
        required=False,
        default=0.9422023296356201,
    )

    parser.add_argument(
        "--scattering",
        dest="scattering",
        type=float,
        help="sets scattering coefficient",
        required=False,
        default=0.9,
    )

    parser.add_argument(
        "--max-order",
        dest="max_order",
        type=int,
        help="sets max reflection order",
        required=False,
        default=9,
    )

    parser.add_argument(
        "--jitter-dims",
        dest="jitter_dims",
        type=float,
        help="sets jitter STD for room dims (no jitter if None)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--jitter-abs",
        dest="jitter_abs",
        type=float,
        help="sets jitter STD for surface absorption coefficients (no jitter if None)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--jitter-scattering",
        dest="jitter_scattering",
        type=float,
        help="sets jitter STD for scattering coefficient (no jitter if None)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--n-rirs",
        dest="n_rirs",
        type=float,
        help="number of RIRs to generate",
        required=False,
        default=20000,
    )

    parser.add_argument(
        "--z-offset",
        dest="z_offset",
        type=float,
        help="offset from ground for rirs",
        required=False,
        default=5e-2,
    )
    
    parser.add_argument(
        "--blocksize",
        dest="blocksize",
        type=int,
        help="number of rirs to compute per job",
        required=False,
        default=100,
    )
    
    parser.add_argument(
        "--n-jobs",
        dest="n_jobs",
        type=int,
        help="number jobs for computing rirs",
        required=False,
        default=16,
    )

    args = parser.parse_args()  


    default_conf_dict = {
        'room' : {
            'dim' : args.room_dims,
            'ceil_offset' : args.ceil_offset,
            'wall_abs' : args.wall_abs,
            'flor_abs' : args.flor_abs,
            'ceil_abs' : args.ceil_abs,
            'scattering' : args.scattering,
            'max_order' : args.max_order,
            'sr' : 125000,
            'air_abs' : (scaled_coeffs, center_freqs),

            'jitter' : {
                'dims' : args.jitter_dims,
                'abs' : args.jitter_abs,
                'scattering' : args.jitter_scattering
            }
        },

        'mics' : {
            'mic_pos' : default_mic_pos,
            'mic_dir' : default_mic_dir, #if none, point to center, otherwise, specify x,y,z
            'mic_pattern' : "SUBCARDIOID", #specify directivity pattern
            'mic_diam' : 0.036, #from aramis' experiments
        },

        'srcs' : {
            'n_src' : args.n_rirs, #specified per aramis' experiments
            'src_pattern' : "OMNI", #specify directivity pattern
            'z_offset': args.z_offset #offset from ground for sources
        },

        'seed' : 5042024

    }

        
    if args.load_yaml is not None:
        with open(args.load_yaml, 'r') as f:
            conf_dict = yaml.load(f, yaml.Loader)
    else:
        conf_dict = default_conf_dict
    
    if args.save_yaml is not None:
        with open(args.save_yaml, 'w') as f:
            yaml.dump(conf_dict, f)
    
    if args.save_dir is None:
        args.save_dir = args.save_path.split('.')[0]
        print(f"Storing parts at {args.save_dir}")
        
    if os.path.exists(args.save_dir):
        rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    
    if args.compute_rirs:
        n_rir = int(conf_dict['srcs']['n_src'])
        block_size = args.blocksize
        
        p_args = [[conf_dict, block_size, os.path.join(args.save_dir, f'piece{i}.part')] for i in range(n_rir//block_size)]
        if n_rir%block_size>0:
            p_args.append([conf_dict, n_rir%block_size, os.path.join(args.save_dir, f'piece{n_rir//block_size}.part')])
        _ = store_rir_part(*p_args[0])
        piece_paths = pqdm(p_args, store_rir_part, n_jobs=args.n_jobs, argument_type='args')
        print("Merging parts")
        merge_rir_dbs(args.save_path, piece_paths)
        print(f"Stored at {args.save_path}")
        rmtree(args.save_dir)