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
Idea of this script: Design the coordinate transform which maps the coordinates from the HEE data to a space such that the lanes are all straight etc.

BACKGROUND INFO, INSIGHTS ABOUT DATA AND LANE LINE KML:

* In the kml file, the formal seems to be triples of "longitude, latitude, 0.0", separated by space. But get_limiting_polygons_from_kml() transforms them to (lat,long) pairs
* so stick with (lat,long) convention here. where long corresponds to x axes (along the highway) and lat to y

APPROACH IDEA

Extrapolate from the following constraints:
* along the lane borders the y coordinates should be 0, 1, 2, 3
* and the x coordinates should be
  * either simply the original ones (keep-x / keep-long appraoch)
  * or the summed length of the lines contained in the polygones at the lane boundaries
"""


import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import torch
import importlib

import hee_loading_and_preprocessing as sl

import pickle


# DEFINE BASIC CONSTANT OBJECTS LIKE LANE BOUNDARIES


rawest_polygons = sl.get_limiting_polygons_from_kml()

onramp_outer_straightish_line_polygon = rawest_polygons['Enter Lane'][-15:-9]
separating_onramp_highway_line_polygon = rawest_polygons['Lanes North (East to West)'][-16:-1]

# For visualization purposes:
onramp_outerish_line_polygon = rawest_polygons['Enter Lane'][-19:-9]  # not just the straightish piece, but till the interesection with the highway boundary line

defining_limiting_polygons_key_val_pairs = [  # to guarantee the order, starting from outermost (onramp) line
    ('Enter Lane (modified)', onramp_outer_straightish_line_polygon),
    ('Lanes North (East to West) (modified)', separating_onramp_highway_line_polygon),
    ('Dashed Line 1/2', rawest_polygons['Dashed Line 1/2']),
    ('Dashed Line 2/3', rawest_polygons['Dashed Line 2/3'])
]

visualization_limiting_polygons_key_val_pairs = [  # to guarantee the order, starting from outermost (onramp) line
    ('Enter Lane (modified)', onramp_outerish_line_polygon),
    ('Lanes North (East to West) (modified)', separating_onramp_highway_line_polygon),
    ('Dashed Line 1/2', rawest_polygons['Dashed Line 1/2']),
    ('Dashed Line 2/3', rawest_polygons['Dashed Line 2/3'])
]


# HELPER METHODS


def transform_polygon_const_lat(orig_polygon_along_highway, const_transf_lat, min_long, max_long):
    """
    Simplest transformation: set lat to const and keep long.
    """

    transf_polygon = []

    for orig_point in orig_polygon_along_highway:

        orig_lat = orig_point[0]
        orig_long = orig_point[1]

        transf_long = orig_long
        transf_lat = const_transf_lat

        transf_point = [transf_lat, transf_long]

        transf_polygon.append(transf_point)

    return transf_polygon


def produce_defining_polygons():
    return {elem[0]: elem[1] for elem in defining_limiting_polygons_key_val_pairs}


# PRESTEPS -- GENERATE THE GROUND TRUTH "LABELS" FOR THE LANE LINE POLYGONS


orig_polygons_dict = produce_defining_polygons()

all_polygon_points = np.concatenate(np.array(list(orig_polygons_dict.values())))

min_lat = all_polygon_points[:, 0].min()
max_lat = all_polygon_points[:, 0].max()
min_long = all_polygon_points[:, 1].min()
max_long = all_polygon_points[:, 1].max()

num_lane_markings = len(orig_polygons_dict)
transf_polygons = []
for lane_marking_num_starting_from_outer, polygon in enumerate(orig_polygons_dict.values()):  # the polygons are ordered in such a way that there index actually coincides with the target transformed y coord.
    transf_polygons.append(np.array(transform_polygon_const_lat(polygon, num_lane_markings-lane_marking_num_starting_from_outer-1, min_long, max_long)))


# MAIN TRANSFORM FUNCTIONALITY


def normalize_longs(points):
    return np.array([[point[0], (point[1] - min_long) / (max_long - min_long)] for point in points])


def normalize_points(points):
    return np.array(
        [[(point[0] - min_lat) / (max_lat - min_lat), (point[1] - min_long) / (max_long - min_long)] for point in
         points])


def generate_transform_without_normalization():

    # FITTING THE SPLINES

    # Idea for keep-long approach:
    # Just need to fit w.r.t first output dimension

    predictors = np.concatenate(np.array(list(correct_limiting_polygons_dict(orig_polygons_dict).values())))
    predictors = normalize_longs(predictors)  # it's a bit hacky to do normalization here and not, say, right from the beginning; but easier
    predicteds = np.concatenate(transf_polygons)[:, 0]

    transf_first = interpolate.interp2d(predictors[:, 0], predictors[:, 1], predicteds, kind='linear')

    def transf(point_lat_long):
        return transf_first(point_lat_long[0], point_lat_long[1]), point_lat_long[1]

    return transf, (predictors, predicteds)


def generate_transform():
    """Just add the prior longi normalization."""

    transf_without_normalization, __ = generate_transform_without_normalization()

    def transf(point):
        return transf_without_normalization(normalize_longs([point])[0])

    return transf


def rotate_by_pi(point):
    return 1 - point[0], 4 - point[1]


def correct_limiting_polygons_dict(limiting_polygons_dict, lat_offset=0.000008, long_offset=-0.000048):
    """NB: the default offsets are roughly estimated by visually fitting the lefter part of the overall view between markings and trajs."""

    res = {}

    def translate_point(point):
        return point[0] + lat_offset, point[1] + long_offset  # based on rough visual guesses

    for key, val in limiting_polygons_dict.items():
        res[key] = [translate_point(point) for point in val]

    return res


# VIS TOOLS


def visualize_indiv_trajs(trajs, limiting_polygons=None, rotate=True):  # , coordinate_system='straight_space'):

    transf = generate_transform()

    corrected_limiting_polygons_key_val_pairs = list(correct_limiting_polygons_dict(dict(visualization_limiting_polygons_key_val_pairs)).items())

    limiting_polygons = [list(map(transf, polygon)) for polygon in
                         np.array(corrected_limiting_polygons_key_val_pairs)[:,
                         1]] if limiting_polygons is None else limiting_polygons

    if rotate:
        limiting_polygons = [[rotate_by_pi(point) for point in polygon] for polygon in limiting_polygons]

    fig = plt.figure()
    ax = fig.add_subplot()

    for limiting_polygon in limiting_polygons:
        ax.plot(np.array(limiting_polygon)[:, 1], np.array(limiting_polygon)[:, 0], c='b')

    for traj in trajs:
        ax.scatter(np.array(traj)[:, 1], np.array(traj)[:, 0], c='r')
        ax.scatter(np.array(traj)[0:1, 1], np.array(traj)[0:1, 0], c='b')  # start point

    plt.show()
