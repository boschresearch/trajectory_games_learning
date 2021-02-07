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
Idea of this script: Collection of preliminary tools for loading, preprocessing and extracting relevant situations in the HEE data sets.
"""


import numpy as np
import pandas as pd
import shapely
import shapely.geometry
from xml.dom import minidom
import functools
import warnings


# PATHS:


main_data_dir = '../data_test/'  # here you have to put the dir where the HEE data and information lies. Details of where to get the data can be found in the paper "Learning game-theoretic models of multiagent trajectories using implicit layers" (https://arxiv.org/abs/2008.07303)
default_dir = main_data_dir  # for data.csv file
kml_dir = main_data_dir  # for map.kml file


# LOAD (AND SAVE) DATA


def load_data(dir=default_dir, type='pandas', formatting=None):
    """
    Load data from CSV file that contains the raw HEE data.

    NB: the nonunique_objectid_to_unique is a bit of a hack to deal with the fact that in the new data, objectid alone is suddenly not globally unique anymore, but only videoid and objectid together, while the functinality was written under the assumption of a unique objectid.
    """

    def videoid_objectid_to_unique(videoid, objectid):

        return videoid*1000 + objectid

    def nonunique_objectids_to_unique(data):
        """
        This is a hack to overcome the mentioned problem. It assumes that the frames are well ordered along the iloc. Not entirely sure if we get downstream problems by this though.
        """

        col_loc = data.columns.get_loc('objectid')  #('frameid')
        print('Start resetting objectid. This may take a while. Total length of rawest data:', len(data))
        for i in range(len(data)):
            videoid, objectid = data.iloc[i][['videoid', 'objectid']]
            unique_objectid = videoid_objectid_to_unique(int(videoid), int(objectid))
            data.iat[i, col_loc] = unique_objectid
            if i % 10000 == 9999:
                print('Resetting objectid at index', i, 'with videoid', videoid, 'and old objectid', objectid, 'to', unique_objectid)

        return data

    if type == 'pandas':

        with open(dir + 'objectposition.csv') as file:
            data = pd.read_csv(file, delimiter='\t', header=0)

        if formatting == 'turn_nonunique_objectids_to_unique':
            data = nonunique_objectids_to_unique(data)

    else:
        with open(dir + 'objectposition.csv') as file:
            data = np.loadtxt(file, delimiter='\t', skiprows=1)

    return data


# PREPROCESS DATA


def raw_to_objectwise_pandas(raw_data, drop_empty_trajs=False):
    """
    Take the raw data from objectposition.csv and turn it into a dict of (pandas) trajectories, one for each object (=vehicle).
    """

    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError('raw_data has to be a pandas.DataFrame')

    min_frameid = int(raw_data['frameid'].min())
    max_frameid = int(raw_data['frameid'].max())

    objectid_indices_pairs = raw_data.groupby('objectid').groups.items()  # it's a dict
    objectids = []
    objectwise_data = []
    objectid_to_index = {}

    length = len(objectid_indices_pairs)

    for i, objectid_indices in enumerate(objectid_indices_pairs):

        print('At i=' + str(i) + ' of ' + str(length))

        objectid = objectid_indices[0]
        indices = objectid_indices[1]

        traj_orig = raw_data[['frameid', 'latitude', 'longitude']].loc[indices].values
        traj_as_frame_position_pairs = [[int(elem[0]), (elem[1], elem[2])] for elem in traj_orig]
        frameids = [int(elem[0]) for elem in traj_orig]
        positions = [[elem[1], elem[2]] for elem in traj_orig]
        traj_as_dataframe_with_frameid_as_col = pd.DataFrame(data=traj_as_frame_position_pairs, columns=['frameid', 'latitude/longitude']).sort_values(by='frameid')
        traj_as_dataframe = pd.DataFrame(data=positions, columns=['latitude', 'longitude'], index=frameids).sort_index()

        objectids.append(objectid)
        objectwise_data.append({'objectid': objectid, 'traj': traj_as_dataframe})
        objectid_to_index[objectid] = i

    if drop_empty_trajs:
        i = 0
        while i < length:
            if i < len(objectwise_data):
                if objectwise_data[i] == []:
                    objectwise_data.pop(i)
                else:
                    i += 1
            else:
                break

    return objectwise_data, objectid_to_index


# FILTERING FOR SCENES ETC:


def position_inside_polygon(pos, polygon):

    lat = pos[0]
    long = pos[1]

    polygon_as_shapely = shapely.geometry.polygon.Polygon(polygon)
    pos_as_shapely = shapely.geometry.Point(pos)

    return polygon_as_shapely.intersects(pos_as_shapely)


def extract_interesting_scenes(raw_data, objectwise_data, objectid_to_index, cap_frameid=None, end_scene_upon='car_leaving_video', incl_early_segment=True):
    """
    Extract suitable merging scenes. E.g. two-cars (as used in the experiment in the paper) with no other interfering cars nearby.
    """

    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError('raw_data has to be a pandas.DataFrame')

    min_frameid = int(raw_data['frameid'].min())
    max_frameid = int(raw_data['frameid'].max())

    def get_objectids_for_frameids(objectwise_data):
        objectids_for_frameids = {frameid: [] for frameid in range(min_frameid, max_frameid + 1)}
        for objectid_traj in objectwise_data:
            objectid = objectid_traj['objectid']
            traj = objectid_traj['traj']
            for frameid in traj.index: # traj['frameid']:
                objectids_for_frameids[frameid].append(objectid)
        return objectids_for_frameids

    objectids_for_frameids = get_objectids_for_frameids(objectwise_data)

    def get_all_objectids_for_a_frameid(frameid, objectids_for_frameids=objectids_for_frameids):
        return objectids_for_frameids[frameid]

    # Idea for the following: How to abstractly define a scence? Define when scene starts and when it ends. And which of the objects are relevant

    def get_all_objectids_in_polygon_in_frame(frameid, objectids, polygon, objectid_to_index=objectid_to_index):
        objectids_in_polygon = []
        for objectid in objectids:
            object_index = objectid_to_index[objectid]
            if position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon):
                objectids_in_polygon.append(objectid)
        return objectids_in_polygon

    def check_cond_start_and_return_objectids(frameid, objectwise_data, objectids_in_frame, objectid_to_index=objectid_to_index):

        objectids_on_onramp = get_all_objectids_in_polygon_in_frame(frameid, objectids_in_frame, polygon_around_onramp)

        objectids_on_right_highway_lane = []
        objectids_on_relevant_highway_stripe = []
        objectids_on_onramp = []
        objectids_on_relevant_onramp_stripe = []
        cond_some_obj_first_time_in_early_onramp_cutthrough = False

        for objectid in objectids_in_frame:

            object_index = objectid_to_index[objectid]

            cond_on_right_highway_lane = (position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_north_of_right_dashed_line) and
                                          position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_around_highway_lanes))  # problem with polygon_around_highway_lanes is that it is too restrictive on the northern side, it throws out many points that should actually be on the highway
                                          # not position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_around_onramp) and
                                          # not position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_around_offramp))
            cond_on_relevant_highway_stripe = cond_on_right_highway_lane and position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_relevant_highway_stripe)
            cond_obj_first_time_in_early_onramp_cutthrough = position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_early_onramp_cutthrough) and not position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid-1], polygon_early_onramp_cutthrough)  # this should make sure that "each actual scene is only extracted once"
            cond_on_onramp = position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_around_onramp)
            cond_on_relevant_onramp_stripe = cond_on_onramp and position_inside_polygon(objectwise_data[object_index]['traj'].loc[frameid], polygon_relevant_onramp_stripe)

            if cond_on_right_highway_lane: objectids_on_right_highway_lane.append(objectid)

            if cond_on_relevant_highway_stripe: objectids_on_relevant_highway_stripe.append(objectid)

            if cond_on_onramp: objectids_on_onramp.append(objectid)

            if cond_on_relevant_onramp_stripe: objectids_on_relevant_onramp_stripe.append(objectid)

            if cond_obj_first_time_in_early_onramp_cutthrough:
                cond_some_obj_first_time_in_early_onramp_cutthrough = True
                objectid_first_time_in_early_onramp_cutthrough = objectid

        cond_start = (cond_some_obj_first_time_in_early_onramp_cutthrough and
                      len(objectids_on_relevant_onramp_stripe) == 0 and  # this does not collide with some_obj_first_time_in_early_onramp_cutthrough because the given onramp polygon is actually completely besides the actual early onramp
                      len(objectids_on_relevant_highway_stripe) == 1)

        relevant_objectids = {} if not cond_start else {'objectid_onramp': objectid_first_time_in_early_onramp_cutthrough, 'objectid_highway': objectids_on_relevant_highway_stripe[0]}

        return cond_start, relevant_objectids

    # Now loop over frames and see which one meets the starting condition of an interesting scence:

    to_frameid = cap_frameid if not cap_frameid is None else max_frameid+1
    scenes = []

    for frameid in range(min_frameid, to_frameid):

        print('Going through frames for starting point: frameid=' + str(frameid) + '. (max_frameid=' + str(max_frameid) + '.)')

        objectids_in_frame = get_all_objectids_for_a_frameid(frameid)

        cond_start, relevant_objectids = check_cond_start_and_return_objectids(frameid, objectwise_data, objectids_in_frame, objectid_to_index)

        if cond_start:

            print('Found new starting point at frameid=' + str(frameid))

            scene_start_frameid = frameid

            if end_scene_upon == 'car_leaving_video':

                leave_frameids = []
                for objectid in relevant_objectids.values():
                    object_index = objectid_to_index[objectid]
                    leave_frameids.append(objectwise_data[object_index]['traj'].index[-1])
                scene_end_frameid = min(leave_frameids) if len(leave_frameids) > 0 else None

            else:
                scene_end_frameid = None

            scene = merge_dicts({'scene_start_frameid': scene_start_frameid, 'scene_end_frameid': scene_end_frameid}, relevant_objectids)

            if incl_early_segment:

                enter_frameids = []
                for objectid in relevant_objectids.values():
                    object_index = objectid_to_index[objectid]
                    enter_frameids.append(objectwise_data[object_index]['traj'].index[0])
                enter_frameid = max(enter_frameids)

                scene['early_start_frameid'] = enter_frameid

            scenes.append(scene)

    print('Finished going through frames for starting point.')

    return scenes


def generate_objectid_to_index(objectwise_data):
    objectid_to_index = {}
    for i, elem in enumerate(objectwise_data):
        objectid_to_index[elem['objectid']] = i
    return objectid_to_index


def scenes_meta_to_trajs(objectwise_data, scenes, objectid_to_index=None, filter_min_prescene_length=None, filter_min_scene_length=None, subsample_step=1):

    objectid_to_index = objectid_to_index if objectid_to_index is not None else generate_objectid_to_index(objectwise_data)

    joint_trajs = []
    scenes_filtered = []

    for scene in scenes:

        scene_start_frameid = scene['scene_start_frameid']
        early_start_frameid = scene['early_start_frameid']
        scene_end_frameid = scene['scene_end_frameid']
        objectid_onramp = scene['objectid_onramp']
        objectid_highway = scene['objectid_highway']

        start = early_start_frameid
        end = scene_end_frameid

        traj_onramp = objectwise_data[objectid_to_index[objectid_onramp]]['traj'].loc[start:end:subsample_step]  # Note that contrary to usual python slices, both the start and the stop are included
        traj_highway = objectwise_data[objectid_to_index[objectid_highway]]['traj'].loc[start:end:subsample_step]

        joint_traj = pd.DataFrame.merge(traj_onramp, traj_highway, left_index=True, right_index=True, how='outer', suffixes=('_onramp', '_highway'))  # , on='frameid'

        cond_prescene = filter_min_prescene_length is None or scene_start_frameid-early_start_frameid>filter_min_prescene_length
        cond_scene = filter_min_scene_length is None or scene_end_frameid-scene_start_frameid>filter_min_scene_length

        if cond_prescene and cond_scene:  # TODO: doing the filtering here is a bit of a hack. maybe do it elsewhere in the future
            joint_trajs.append(joint_traj)
            scenes_filtered.append(scene)
        else:
            pass

    return joint_trajs, scenes_filtered


def filter_start_point(obj_data):
    """
    Filter for those trajectories, that start at the beginning (in the image) of the on-ramp lane.
    """

    lat_max = 51.061234
    lat_min = 51.061176
    long_min = 13.523211
    long_max = 13.523267

    # NB: The box is based on the FIRST recording we have.
    # NB: The lat/longs of the recordings do have an offset (about one lane width) from the lat/longs one gets from google maps.

    res = []

    for traj in obj_data:
        point = traj[0]
        if lat_min <= point[0] <= lat_max and long_min <= point[1] <= long_max:
            res.append(traj)

    return res


# GET AND DEFINE RELEVANT INFORMATION ABOUT THE LANE MARKINGS AND FILTERING FOR MERGE SCENES:


def get_limiting_polygons_from_kml(dir=default_dir):

    kml_file_path = kml_dir + 'map.kml'

    with open(kml_file_path) as file:
        kml_doc = minidom.parse(kml_file_path)

    placemark_elems = kml_doc.getElementsByTagName('Placemark')
    polygons = {}
    for elem in placemark_elems:
        placemark_name = elem.getElementsByTagName('name')[0].firstChild.nodeValue
        coords_raw_string = elem.getElementsByTagName('coordinates')[0].firstChild.nodeValue
        coords_raw_array = coords_raw_string.split(' ')
        polygon = []
        for coord_triplet_string in coords_raw_array:
            triplet = coord_triplet_string.split(',')
            lat = float(triplet[1])  # lat and long seem to be swapped
            long = float(triplet[0])
            polygon.append((lat, long))
        polygons[placemark_name] = polygon

    return polygons


def generate_max_rectangle_from_min_longitude(min_longitude):
    return [(max_sample_lat, min_longitude), (max_sample_lat, max_sample_long), (min_sample_lat, max_sample_long), (min_sample_lat, min_longitude), (max_sample_lat, min_longitude)]


limiting_polygons = get_limiting_polygons_from_kml()

# KML contains: ['Lanes North (East to West)', 'Exit Lane', 'Enter Lane', 'Enter Lane Line', 'Exit Lane line', 'Dashed Line 1/2', 'Dashed Line 2/3']

polygon_around_onramp = limiting_polygons['Enter Lane']
polygon_around_offramp = limiting_polygons['Exit Lane']
polygon_around_highway_lanes = limiting_polygons['Lanes North (East to West)']
polygon_north_of_right_dashed_line = limiting_polygons['Dashed Line 1/2'] + [(51.060741, 13.518129), (51.060793, 13.525582)] + [limiting_polygons['Dashed Line 1/2'][0]]  # i guess this is the easiest way to get cars on the right highway lane by taking the setminus
polygon_early_onramp_cutthrough = [(51.061001, 13.523225), (51.061001, 13.523275), (51.060991, 13.523275), (51.060991, 13.523225), (51.061001, 13.523225)]

min_sample_lat = 51.060515
max_sample_lat = 51.061690999999996
min_sample_long = 13.51774
max_sample_long = 13.528175

polygon_relevant_highway_stripe = generate_max_rectangle_from_min_longitude(13.525151) #13.524600) #13.523503)
polygon_relevant_onramp_stripe = generate_max_rectangle_from_min_longitude(13.521450)


# OTHER TOOLS:


def merge_dicts(d1, d2):
    return dict(list(d1.items()) + list(d2.items()))
