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
Idea of this script: Example for how to get and visualize suitable merge scene joint trajectories form the HEE data set.
"""


import numpy as np
import pandas as pd
import pickle

import hee_coordinate_transform as ct
import hee_preprocessing as hp


# EXAMPLE LOADING AND VIS JOINT TRAJ


joint_trajs = hp.load_suitable_joint_trajs(cut_off_initial_segment_length=125, filter_for_within_two_lanes=True)
# at the very beginning the onramp traj is sometimes completely screwed, but cut_off_initial_segment_length=125 seems a good number to cutt off

joint_traj = joint_trajs[23]

onramp_traj = joint_traj[:, 0:2]
highway_traj = joint_traj[:, 2:4]

ct.visualize_indiv_trajs([onramp_traj, highway_traj])

input("Press Enter to continue...")