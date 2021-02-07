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
This script runs the evaluation on the test sets using a trained game and a trained subspace predictor.
The trained modles have to be present in the "trained_models" directory. 
"""

import numpy as np
import torch
from tg.potential_game import PotGame

from tg.merging_game import pot_otherv, pot_mergerv, decode_sols, plot_game, GamePredictor,SubspacePredictor,save_trajectories
from tg.merging_game import encode_sols2 as encode_sols
import tg.merging_game

import numpy as np
from data import hee_preprocess2


dataset_name_ssp = "hee-pretrained" # for the pretrained models either "hee-pretrained" or "highD-pretrained"
dataset_name = "hee-pretrained" # for the pretrained models either "hee-pretrained" or "highD-pretrained"

# these parameters have to be the same as used during training of the game and subspace predictor
use_deltav = True
t_hist = 5
spacing = 5
nhidden = 16
scale_accel = 1
game_verbose = True

validate = False
pred_size = 65
velocity = 0
layers = 2

if dataset_name.startswith("highD"):
    from data import highD_preprocessing2 as highD_preprocessing
    tracks = np.array([x[::spacing,:] for idx,x in highD_preprocessing.get_all_merge_tracks(extremefilter=True,mindelta=0,startpos=True,min_length=8*25)])
    tracks = np.array([x[:8*5+1,:] for x in tracks])
    pred_length = 7*5+1   
    eval_points = (np.array([1,2,3,4,5,6,7])*25)//5   
    avgx = np.mean(np.array([np.mean(x[1:,0]-x[0:-1,0]) for x in tracks]))
    scaling = (np.array([1/avgx*3, 1, 1/avgx*3, 1])[None,:])
    tracks = np.array([x*scaling for x in tracks])
    scaling = (1 / scaling)
    nhidden = 16
elif dataset_name.startswith("hee"):
    merge_tracks, scaling_fac = hee_preprocess2.get_all_merge_tracks2(skip=5, return_scaling=True)
    tracks = np.array([x for idx,x in enumerate(merge_tracks)])
    tracks = np.array([x[:8*5+1,:] for x in tracks])
    pred_length = 7*5+1 
    eval_points = (np.array([1,2,3,4,5,6,7])*25)//5
    nhidden = 16
    scaling = (scaling_fac)
else:
    print("Unknown dataset", dataset_name)
   

print("End merger y positions", tracks[:,8*5,3])

print("TRACKS SHAPE", tracks.shape) 
avgx = [np.mean(x[1:,0]-x[0:-1,0]) for x in tracks]

print("Average deltax", avgx, np.mean(np.array(avgx)))


print("Lenght of tracks", len(tracks))

from sklearn.model_selection import KFold
kf = KFold(n_splits=4, random_state=0, shuffle=True)

fold = 0

total_rmsq = np.zeros((pred_length,))
total_count = np.zeros((pred_length,))
total_mae = np.zeros((pred_length,))

for train_idx, test_idx in list(kf.split(tracks))[fold:]:
    fold += 1
    
    val_idx = train_idx[train_idx.shape[0]//3*2:]
    train_idx = train_idx[:train_idx.shape[0]//3*2]

    tracks_train = tracks[train_idx]
    tracks_val = tracks[val_idx]
    tracks_test = tracks[test_idx]

    train_conds = encode_sols(tracks_train,offset = t_hist)
    val_conds = encode_sols(tracks_val,offset = t_hist)
    test_conds = encode_sols(tracks_test, offset = t_hist)

    desired_v = ((tracks_train[0][10,0]-tracks_train[0][0,0])/10)


    train_games = [PotGame(
                  convex_ndims=4*(tracks_train[i].shape[0]-t_hist),
                  player_utilities=(pot_otherv,pot_mergerv)
                  ) for i in range(len(tracks_train))]
    val_games = [PotGame(
                  convex_ndims=4*(tracks_val[i].shape[0]-t_hist),
                  player_utilities=(pot_otherv,pot_mergerv)
                  ) for i in range(len(tracks_val))]
    test_games = [PotGame(
                  convex_ndims=4*(tracks_test[i].shape[0]-t_hist),
                  player_utilities=(pot_otherv,pot_mergerv)
                  ) for i in range(len(tracks_test))]

    thetam = torch.tensor([0., 0, 0, 0, 0, 0,1,1,0]) # is not used - only for instantiation, parameters are later loaded from stored model
    model = GamePredictor(thetam,2, None, t_hist=t_hist, nhidden=nhidden, nlayers = layers, velocity = velocity, use_deltav=use_deltav)
    model.eval()

    ssp = SubspacePredictor(t_hist=t_hist,nhidden=64,layers=1)
    ssp.eval()


    # load pretrained model    
    ssp.load_state_dict(torch.load("trained_models/%s_ssp_spacing%d_fold%d.torch" % (dataset_name_ssp,spacing,fold,)))    
    model.load_state_dict(torch.load("trained_models/%s_game_model_vdelta%d_spacing%d_scaleaccel%d_vel_predsize%d_fold%d.torch" % (dataset_name,use_deltav,spacing,scale_accel,pred_size,fold,)))     
    print("Model parameters", model.theta)
    ssp.eval()
    model.eval()

    pred_test_ss = []
    for i in range(len(tracks_test)):     
        subspace = ssp(torch.tensor(tracks_test[i][:t_hist,:]).float())
        print(i, "before_after", test_conds[i][0], subspace[0],  subspace[1], test_conds[i][1])
        pred_test_ss.append((np.argmax(subspace[0].detach().cpu()),int(subspace[1].detach().cpu())))
        #pred_test_ss.append(test_conds[i][0])

    if True:    
         import importlib
         import tg.merging_game
         importlib.reload(tg.merging_game)

         fold_rmsq = total_rmsq*0
         fold_mae = total_mae*0
         fold_count = total_count*0
         for i in range(len(tracks_test)):
             #if dataset_name.startswith("highD") and fold == 2 and i == 3:
             #  continue # this track unfortunately wasnt filtered out by the data loading filter but is a track of a car that continues driving on the sidelane and is therefore not a mergetrack, was filtered out for all models       
             #if tracks_test[i][-1,3] <= 0.5:
             #  continue # apparently not a merge track :-/
             
             #if fold != 2 or i != 3:
             #  continue

             print("Track", i)
             fold_count += 1
             model.zero_grad()
             print(" gt before/after, t", test_conds[i][0])
             print(" pred before/after, t", pred_test_ss[i])
             print(" history velocity: ", (tracks_test[i][t_hist-1,:]-tracks_test[i][t_hist-2,:])[[0,2]])
             sols = model(None, test_games[i],torch.tensor(tracks_test[i][:t_hist,:]).float(),pred_test_ss[i],test_conds[i][1], validate = validate, verbose=game_verbose, history_size = pred_size)
             
             gt = tracks_test[i][t_hist:]
             save_trajectories("./predictions/%s_vdelta%d_validate%d_fold%d_predsize%d_testtrack%d.pickle" % (dataset_name,use_deltav,validate,fold,pred_size, i), sols, tracks_test[i],t_hist)
             delta = (sols[2][:,:].detach().cpu().numpy()-gt[:,:])*scaling
             lossmaei = np.mean(np.abs(delta), axis=1)*2
             lossmsqei = np.mean(((delta)**2),axis=1)*2
             fold_mae += lossmaei
             fold_rmsq += lossmsqei
             print("  delta", delta[eval_points])
             print("MAE Loss", i, lossmaei[eval_points])
             print("RMSQE Loss", i, lossmsqei[eval_points]**0.5)
             #tg.merging_game.plot_game(sols,test_conds[i][1],gt)
         total_mae += fold_mae
         total_rmsq += fold_rmsq
         total_count += fold_count
         print("Total Fold %d Testloss MAE" % (fold,), (fold_mae/fold_count)[eval_points])        
         print("Total Fold %d Testloss mean MAE" % (fold,), (fold_mae/fold_count)[eval_points].mean())        
         print("Total Fold %d Testloss RMSE" % (fold,), ((fold_rmsq/fold_count)**0.5)[eval_points])
         print("Total Fold %d Testloss mean RMSE" % (fold,), ((fold_rmsq/fold_count)**0.5)[eval_points].mean())

print("Total Testloss MAE", (total_mae/total_count)[eval_points])
print("Total Testloss mean MAE", (total_mae/total_count)[eval_points].mean())
print("Total Testloss RMSE", ((total_rmsq/total_count)**0.5)[eval_points])
print("Total Testloss mean RMSE", ((total_rmsq/total_count)**0.5)[eval_points].mean())
    
