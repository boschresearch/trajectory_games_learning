"""
// Copyright (c) 2020 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Idea of this script: Combine the existing preprocessing steps into a loader. Output: joint trajs that match into our trajectory game learning framework.
"""


import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import torch
import importlib
import sympy  # for symbolic intregration

# this is needed for plotting long/lat coordinates but may be commented when not using it:
#import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt

import hee_loading_and_preprocessing as sl
import hee_coordinate_transform as ct

import pickle


# PATHS ETC.:


main_data_dir = sl.main_data_dir  # here you have to put the dir where the HEE data and information lies; it'll also be used for storing intermediate extractions
extracted_data_dir = main_data_dir + 'extracted_data/'

objectposition_dir = main_data_dir
uniqueobjectided_objectposition_path = extracted_data_dir + 'uniqueobjectided_objectposition.pkl'

dir_pickle_objectwise_data = extracted_data_dir
dir_pickle_scenes = extracted_data_dir
path_pickle_objectwise_data = dir_pickle_objectwise_data + 'objectwise_data.pkl'
path_pickle_scenes = dir_pickle_scenes + 'scenes.pkl'


# MAIN FUNCTION:


def load_suitable_joint_trajs(cut_off_initial_segment_length=300, path_pickle_objectwise_data=path_pickle_objectwise_data, path_pickle_scenes=path_pickle_scenes, filter_for_within_two_lanes=True):

    # LOADING, EXTRACTING:

    # Get raw data:

    # Load original
    raw_data = sl.load_data(dir=objectposition_dir, type='pandas', formatting='turn_nonunique_objectids_to_unique')

    # # Save uniquized -- uncomment if wanted:
    # with open(uniqueobjectided_objectposition_path, 'wb') as file:
    #     pickle.dump(raw_data, file, pickle.HIGHEST_PROTOCOL)

    # # Load uniquized -- uncomment if wanted:
    # with open(uniqueobjectided_objectposition_path, 'rb') as file:
    #     raw_data = pickle.load(file)

    # Extract objectwise:

    # Newly extract:
    objectwise_data, objectid_to_index = sl.raw_to_objectwise_pandas(raw_data)

    # # Save -- uncomment if wanted:
    # with open(path_pickle_objectwise_data, 'wb') as file:
    #     pickle.dump(objectwise_data, file, pickle.HIGHEST_PROTOCOL)

    # # Load presaved -- uncomment if wanted:
    # with open(path_pickle_objectwise_data, 'rb') as file:
    #     objectwise_data = pickle.load(file)
    # objectid_to_index = sl.generate_objectid_to_index(objectwise_data)

    # Extract scenes:

    # Newly extract:
    scenes = sl.extract_interesting_scenes(raw_data, objectwise_data, objectid_to_index)

    # # Save -- uncomment if wanted:
    # with open(path_pickle_scenes, 'wb') as file:
    #     pickle.dump(scenes, file, pickle.HIGHEST_PROTOCOL)

    # # Load presaved -- uncomment if wanted:
    # with open(path_pickle_scenes, 'rb') as file:
    #     scenes = pickle.load(file)

    # COORDINATE-TRANSFORM THE TRAJS (this is necessary because the HEE highway section is slightly bent, while for our paper's theoretical quarantees to hold, we need it to be straight)

    importlib.reload(ct)

    transf_wo_rotation = ct.generate_transform()

    def transf(point):
        return ct.rotate_by_pi(transf_wo_rotation(point))

    def transf_indiv_traj(indiv_traj):
        return [transf(point) for point in indiv_traj]

    filter_min_prescene_length = 10
    filter_min_scene_length = 440
    traj_data, scenes_filtered = sl.scenes_meta_to_trajs(objectwise_data, scenes, filter_min_prescene_length=filter_min_prescene_length, filter_min_scene_length=filter_min_scene_length)

    def scalarize_strange_singletons(traj):  # i dont know why there is a strange singleton as first dimension
        if len(traj[0]) == 4:
            return np.array([[elem[0][0], elem[1], elem[2][0], elem[3]] for elem in traj])
        elif len(traj[0]) == 2:
            return np.array([[elem[0][0], elem[1]] for elem in traj])
        else:
            raise

    def onramp_traj_is_within_two_lanes(traj):
        # Treat onramp and highway car traj differently because onramp trajs sometimes have these weird couple of initial points all over the place
        traj = scalarize_strange_singletons(traj)
        start_late = 100  # because of weird initials
        return traj[start_late:, 0].max() < 0

    def highway_traj_is_within_two_lanes(traj):
        traj = scalarize_strange_singletons(traj)
        return traj[:, 0].max() < 0

    output_traj_data = []
    for joint_traj in traj_data:
        onramp_traj = np.array(transf_indiv_traj(joint_traj[['latitude_onramp', 'longitude_onramp']].values)[cut_off_initial_segment_length:])
        highway_traj = np.array(transf_indiv_traj(joint_traj[['latitude_highway', 'longitude_highway']].values)[cut_off_initial_segment_length:])
        if filter_for_within_two_lanes:
            if onramp_traj_is_within_two_lanes(onramp_traj) and highway_traj_is_within_two_lanes(highway_traj):
                output_traj_data.append(np.concatenate((onramp_traj, highway_traj), axis=1))
        else:
            output_traj_data.append(np.concatenate((onramp_traj, highway_traj), axis=1))

    return [scalarize_strange_singletons(traj) for traj in output_traj_data]
