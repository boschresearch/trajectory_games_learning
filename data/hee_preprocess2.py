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


import numpy as np
import pickle
import tg.merging_game
from tg.merging_game import encode_sols2, plot_game
import torch
    
def get_all_merge_tracks2(skip=5, return_scaling=False):
    try:
        dic = pickle.load(open("data/hee_tracks.pickle","rb"))
        tracks = dic["tracks"]
        scaling = dic["scaling"]
        print("Loaded tracks from file")
    except:
        from hee_loading_and_preprocessing import hee_coordinate_transform as ct
        from hee_loading_and_preprocessing import hee_preprocessing as hp

        tracks = hp.load_suitable_joint_trajs(cut_off_initial_segment_length=125, filter_for_within_two_lanes=True)
        # at the very beginning the onramp traj is sometimes completely screwed, but cut_off_initial_segment_length=125 seems a good number to cutt off

        # reorder dimensions
        tracks = [ np.concatenate([tr[:,3:4],tr[:,2:3]+1.5,tr[:,1:2],tr[:,0:1]+1.5],axis=1) for tr in tracks]

        min_length = 1e9

        for tn, tr in enumerate(tracks):
            inlane = np.where(tr[:,3] >= -0.2, 1, 0)
            inlaneallafter = np.cumprod(inlane[::-1])[::-1]
            firstind = np.where(inlaneallafter == 1)[0][0]
            tracks[tn] = tr[firstind::skip,:]
            min_length = min(min_length, tracks[tn].shape[0])


        first_merge_ind = 1e3
        for i,x in enumerate(tracks):
            ind = np.where(x[:,3]>0.5)[0]
            if len(ind) > 0:
                print(i,ind[0]/5,len(ind))
                first_merge_ind = min(first_merge_ind,ind[0])
            else:
                print(i,"No merge track")    
        print("FIRST MERGE IND", first_merge_ind)

        tracks = [ tr[first_merge_ind-6:,:] for tr in tracks]

        ntracks = []
        for i, tr in enumerate(tracks):
            if len(tr) >= 8*5+1:
                ntracks.append(tr)
        tracks = ntracks   
    
        # rescale x dimensions and substract lane end from x
        avgx = np.mean(np.array([np.mean(x[1:,0]-x[0:-1,0]) for x in tracks]))
        scaling = 1/avgx*3 # 3 is the average deltax in highD
        tracks = [ np.concatenate([tr[:,0:1]*scaling - 3.7*scaling,tr[:,1:2],tr[:,2:3]*scaling - 3.7*scaling,tr[:,3:4]],axis=1) for tr in tracks]  

        pickle.dump({"tracks": tracks, "scaling" : scaling}, open("hee_tracks.pickle","wb"))
    
        print("Average x", avgx, "Scaling", scaling)
        print("min_length", min_length)
        
    if return_scaling:
        scalingv = np.ones((1,4))
        scalingv[0,0] = 1/scaling*1043
        scalingv[0,2] = 1/scaling*1043
        return tracks, scalingv
    else:
        return tracks    
    
    
    
    
def get_toy_tracks(skip=5, return_scaling=False):
    tracks,scaling = get_all_merge_tracks2(skip=skip, return_scaling = True)
    
    print("Toy TRACKS:")
    first_merge_ind = 1e3
    for i,x in enumerate(tracks):
        ind = np.where(x[:,3]>0.5)[0]
        if len(ind) > 0:
            print(i,ind[0]/5,len(ind))
            first_merge_ind = min(first_merge_ind,ind[0])
        else:
            print(i,"No merge track")    
            
    xo = tracks[1]
    
    indam = np.where(xo[:,3] > 0)[0][20]
    xam = xo[indam,2]
    
    x1 = xo.copy()
    x1[:,0] = 0#xam
    x1[:,1] = 1
    

    x2 = xo.copy()
    x2[:,0] = 0
    x2[:5,0] = -np.arange(5)[::-1]
    x2[:,1] = 1
    
    #x2 = xo.copy()
    #dxo = xo[:,0] - xo[0,0]
    #x2[:,0] = xo[0,0]+3*dxo
    
    if return_scaling:
        return [x1,x2], scaling
    else:
        return [x1,x2]
    

