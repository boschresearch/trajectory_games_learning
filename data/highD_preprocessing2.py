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

import os
import sys
import pickle
import argparse

from highD.Python.src.data_management.read_csv import *


def create_args(path = "/data/BoschData/highD", scene = "60"):
    
    parser = {}
    # --- Input paths ---

    parser["input_path"] = "%s/data/%s_tracks.csv" % (path,scene)
    parser["input_static_path"] ="%s/data/%s_tracksMeta.csv"  % (path,scene)
    parser["input_meta_path"] ="%s/data/%s_recordingMeta.csv"  % (path,scene)
    parser["pickle_path"] ="%s/data/%s.pickle" % (path,scene)
    parser["background_image"] ="%s/data/%s_highway.jpg" % (path,scene)

    # --- I/O settings ---
    parser["save_as_pickle"] = True
    parsed_arguments = parser
    return parsed_arguments



def read_data(path = "/data/BoschData/highD", scene = "60"):
    created_arguments = create_args(path, scene)
    
    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if os.path.exists(created_arguments["pickle_path"]):
                with open(created_arguments["pickle_path"], "rb") as fp:
                    tracks = pickle.load(fp)
                print("Found pickle file {}.".format(created_arguments["pickle_path"]))
    else:
                print("Pickle file not found, csv will be imported now.")
                tracks = read_track_csv(created_arguments)
                print("Finished importing the pickle file.")

    if created_arguments["save_as_pickle"] and not os.path.exists(created_arguments["pickle_path"]):
                print("Save tracks to pickle file.")
                with open(created_arguments["pickle_path"], "wb") as fp:
                    pickle.dump(tracks, fp)

            # Read the static info
    try:
                static_info = read_static_info(created_arguments)
    except:
                print("The static info file is either missing or contains incorrect characters.")
                sys.exit(1)

            # Read the video meta
    try:
                meta_dictionary = read_meta_info(created_arguments)
    except:
                print("The video meta file is either missing or contains incorrect characters.")
                sys.exit(1)

    if tracks is None:
                    print("Please specify the path to the tracks csv/pickle file.")
                    sys.exit(1)
    if static_info is None:
                    print("Please specify the path to the static tracks csv file.")
                    sys.exit(1)
    if meta_dictionary is None:
                    print("Please specify the path to the video meta csv file.")
                    sys.exit(1)
    return tracks, static_info, meta_dictionary



def get_merge_tracks(path = "/data/BoschData/highD", scene = "60", extremefilter = True, mindelta = 200):
    tracks, static_info, meta = read_data(path, scene)
    print(meta)
    upper_laney = meta["upperLaneMarkings"]
    lower_laney = meta["lowerLaneMarkings"]
    
    # filter tracks on merging lane
    merge_track_ids = []
    for i, t in enumerate(tracks):
        if t["laneId"][0] == 2:
            if (t["laneId"][:]<=3).all() or extremefilter is False:
                merge_track_ids.append(i)
        if t["id"] == 48:
            print("laneIds",t["laneId"])
            
    print("len merge_track_ids", len(merge_track_ids))        
    # filter tracks which have a car in front or back after merging
    merge_track_interaction_ids = []
    merge_track_t = []
    other_track_interaction_ids = []
    other_track_t_offset = []
    other_track_prec = []
    
    for i in merge_track_ids:
        t = tracks[i]
        merge_t = -1
        for j, lid in enumerate(t["laneId"]):
            if lid == 3:
                merge_t = j
                break
        merge_track_t.append(merge_t)
        
        def selector(a,aall,b,ball):
            if extremefilter:
                #if a is True and b is True:
                #    return False
                #elif a is False and b is False:
                #    return False
                #elif a is True:
                #    return aall
                #elif b is True:
                #    return ball
                pass
            return (a and aall)or(b and ball)
            
#        if (t["precedingId"][merge_t] != 0 ) or (t["followingId"][merge_t] != 0 ) :
        if selector(t["precedingId"][merge_t] != 0,((tracks[t["precedingId"][merge_t]-1]["laneId"] == 3).all()),t["followingId"][merge_t],( (tracks[t["followingId"][merge_t]-1]["laneId"] == 3).all())):

            frame = t["frame"][merge_t]
            
            xposf = t["bbox"][merge_t][0]            
            xposb = t["bbox"][merge_t][0]  -t["bbox"][merge_t][2] #corner - width
            
            xpos_prec = 1e9
            xpos_fol = 1e9
            
            if t["id"] == 181:
                print("181 preceding:",t["precedingId"][merge_t])
            
            if t["precedingId"][merge_t] != 0:
                to = tracks[t["precedingId"][merge_t]-1]
                j = np.searchsorted(to["frame"], frame)
                xpos_prec = to["bbox"][j][0] - to["bbox"][j][2] #corner - width
                t_offset_prec = j - merge_t

            if t["followingId"][merge_t] != 0:
                to = tracks[t["followingId"][merge_t]-1]
                j = np.searchsorted(to["frame"], frame)
                xpos_fol = to["bbox"][j][0]
                t_offset_fol = j - merge_t
            
            # select vehicle with smaller distance
            if abs(xpos_prec - xposf) < abs(xpos_fol - xposb) and (abs(xpos_fol - xposb) > mindelta or not extremefilter):
                print("neighboring vehicle found", i)
                merge_track_interaction_ids.append(i+1)
                other_track_interaction_ids.append(t["precedingId"][merge_t])
                other_track_t_offset.append(t_offset_prec)
                other_track_prec.append(True)
            elif (abs(xpos_prec - xposf) > mindelta or not extremefilter):
                print("neighboring vehicle found", i)
                merge_track_interaction_ids.append(i+1)
                other_track_interaction_ids.append(t["followingId"][merge_t])
                other_track_t_offset.append(t_offset_fol)
                other_track_prec.append(False)

    print("len merge_track_interaction_ids", len(merge_track_interaction_ids))        
                
                
    xs = []
    
    for i in range(len(merge_track_interaction_ids)):
        t = tracks[merge_track_interaction_ids[i]-1]
        to = tracks[other_track_interaction_ids[i]-1]
        merge_t = merge_track_t[i]
        offset_t = other_track_t_offset[i]
        prec = other_track_prec[i]
        
        min_t, max_t = max(0,0-offset_t),min(t["bbox"].shape[0],to["bbox"].shape[0]-offset_t)
        
        print("mint,maxt",min_t,max_t,offset_t)
        
        if (max_t - min_t) >= 77:
            if t["id"] == 181:
                print("181 interaction id:",to["id"])    
                print("other",to["bbox"][:,1])
                print("merger", t["bbox"][:,1])

            if max_t > min_t+10:
                x = np.zeros((max_t-min_t,4))

                for j in range(min_t,max_t):
                    assert to["frame"][j+offset_t] == t["frame"][j] 
                    if prec:
                        x[j-min_t,0] = to["bbox"][j+offset_t][0] - to["bbox"][j+offset_t][2]
                        x[j-min_t,1] = to["bbox"][j+offset_t][1] + to["bbox"][j+offset_t][3]/2          
                        x[j-min_t,2] = t["bbox"][j][0]
                        x[j-min_t,3] = t["bbox"][j][1] +  t["bbox"][j][3]/2                       
                    else:
                        x[j-min_t,0] = to["bbox"][j+offset_t][0]
                        x[j-min_t,1] = to["bbox"][j+offset_t][1] + to["bbox"][j+offset_t][3]/2
                        x[j-min_t,2] = t["bbox"][j][0] - t["bbox"][j][2]
                        x[j-min_t,3] = t["bbox"][j][1] + t["bbox"][j][3]/2                       

                # reverse x direction
                x[:,0] = -x[:,0]
                x[:,2] = -x[:,2]


                # normalize y position
                x[:,1] -= upper_laney[1]
                x[:,3] -= upper_laney[1]
                x[:,1] /= (upper_laney[2]-upper_laney[1])
                x[:,3] /= (upper_laney[2]-upper_laney[1])
                x[:,1] += 0.5
                x[:,3] += 0.5
                print("Y scaling", (upper_laney[2]-upper_laney[1]))
                xs.append(((scene, t["id"],to["id"]),x))
    print("Num final tracks", len(xs))
    return xs

def get_all_merge_tracks(extremefilter=True, mindelta=200, startpos=False, t_hist=25, min_length=6*25):
    scenes = ["58", "59", "60"]
    x = []
    for s in scenes:
        xs = get_merge_tracks(scene=s,extremefilter=extremefilter, mindelta=mindelta)
        x.append(np.array(xs))
    x = np.concatenate(x,axis=0)
    if startpos:
        xnew = []
        startpos = -325
        for i in range(len(x)):
            firstind = np.where(x[i][1][:,2] >= startpos)[0][0]
            print("firstind",i,firstind,x[i][1][0,2])
            if firstind > 0 and x[i][1][firstind+t_hist,3] < 0.4:
                xn = x[i][1][firstind:,:]
                xnew.append((x[i][0],xn))
        x = xnew        
    # filter min_length
    xnew = []
    for i in range(len(x)):
        if x[i][1].shape[0] > min_length:
            xnew.append(x[i])
    x = xnew
    import pickle
    xids = [t[0] for t in x]
    pickle.dump(xids, open("highd_merge_tracks.pickle", "wb"))
    return np.array(x)



    
    
def get_toy_tracks(extremefilter=True, mindelta=200, startpos=False, t_hist=25, min_length=6*25):
    tracks = get_all_merge_tracks(extremefilter, mindelta, startpos, t_hist, min_length)    
    xo = tracks[0][1]
    
    indm = np.where(xo[:,3] > 0.5)[0][0] # index of merge
    indam = np.where(xo[:,3] > 0.5)[0][20] # some index after merge
    
    xam = xo[indam,2]
    
    x1 = xo.copy()
    x1[:,2] += 50
    x1[:,0] = xam
    x1[:,1] = 1
    
    
    x2 = xo.copy()
    dxo = xo[:,0] - xo[0,0]
    x2[:,0] = xo[indm-5,0]+2*dxo
    

    return [x1,x2]
    
    