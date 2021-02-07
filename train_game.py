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
This script trains a game model for a merging scenario. The trained model includes the preference revelation net. The trained model is saved in the directory "trained_models" with a filename starting with the dataset-name.
"""
import numpy as np
import torch
from tg.potential_game import PotGame

from tg.merging_game import pot_otherv, pot_mergerv, decode_sols, plot_game, GamePredictor,SubspacePredictor
from tg.merging_game import encode_sols2 as encode_sols

import tg.merging_game

import numpy as np
from data import hee_preprocess2

from torch.multiprocessing import Pool, Process


use_deltav = True  # this indicates wheter the tgl-d method is used
dataset_name = "hee-mytest"
spacing = 5
nhidden = 16
layers = 2
t_hist = 5



scaling = torch.tensor(np.array([1., 1, 1, 1])[None,:])
scale_accel=1
velocity = 0
pred_size = 65
 
if dataset_name.startswith("highD"):
    from data import highD_preprocessing2 as highD_preprocessing
    tracks = np.array([x[::spacing,:] for idx,x in highD_preprocessing.get_all_merge_tracks(extremefilter=True,mindelta=0,startpos=True,min_length=8*25)])
    tracks = np.array([x[:8*5+1,:] for x in tracks])
    pred_length = 7*5+1   
    eval_points = (np.array([1,2,3,4,5,6,7])*25)//5   
    avgx = np.mean(np.array([np.mean(x[1:,0]-x[0:-1,0]) for x in tracks]))
    scaling = (np.array([1/avgx*3, 1, 1/avgx*3, 1])[None,:])
    tracks = np.array([x*scaling for x in tracks])
    scaling = torch.tensor(1 / scaling)
    nhidden = 16
elif dataset_name.startswith("hee"):
    merge_tracks, scaling_fac = hee_preprocess2.get_all_merge_tracks2(skip=5, return_scaling=True)
    tracks = np.array([x for idx,x in enumerate(merge_tracks)])
    tracks = np.array([x[:8*5+1,:] for x in tracks])
    pred_length = 7*5+1 
    eval_points = (np.array([1,2,3,4,5,6,7])*25)//5
    nhidden = 16
    scaling = torch.tensor(scaling_fac)
else:
    print("Unknown dataset", dataset_name)
    raise Exception()
    
from sklearn.model_selection import KFold
kf = KFold(n_splits=4, random_state=0, shuffle=True)
print("Total tracks", len(tracks))

def train_fold(fold):
    train_idx, test_idx = list(kf.split(tracks))[fold]
    fold += 1
    
    
    import sys
    lf = open("logs/%s_train_game_model_vdelta%d_spacing%d_vel_predsize%d_fold%d.log" % (dataset_name,use_deltav,spacing,pred_size,fold,), "a", buffering=1)
    print("see", "logs/%s_train_game_model_vdelta%d_spacing%d_vel_predsize%d_fold%d.log" % (dataset_name,use_deltav,spacing,pred_size,fold,), "for logging")
    sys.stdout = lf
    sys.stderr = lf
    
    print("Working on Fold", fold)
    
    
    
    val_idx = train_idx[int(train_idx.shape[0]*0.6):]
    train_idx = train_idx[:int(train_idx.shape[0]*0.6)]

    tracks_train = tracks[train_idx]
    tracks_val = tracks[val_idx]
    tracks_test = tracks[test_idx]


    train_conds = encode_sols(tracks_train,offset = t_hist)
    val_conds = encode_sols(tracks_val,offset = t_hist)
    test_conds = encode_sols(tracks_test, offset = t_hist)


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

    
    #thetam = torch.tensor([0., 0, 0, 0, -1, 0,1,1,0])
    thetam = torch.tensor([0., 0, 0, 0, -1, 0,2,1,0])
    model = GamePredictor(thetam,None,None, t_hist=t_hist, nhidden=nhidden, nlayers = layers, velocity = velocity, use_deltav = use_deltav)

    import tg.sdlbfgs
    lbfgs_g = tg.sdlbfgs.SdLBFGS(model.parameters(), lr = 0.7, history_size=10, max_iter=10)


    
    
    def val_test_loss():
        model.eval()
        val_loss = 0
        for i in range(len(tracks_val)):
            model.zero_grad()
            sols = model(None,val_games[i],torch.tensor(tracks_val[i][:t_hist,:]).float(),val_conds[i][0],val_conds[i][1], history_size = pred_size)

            y = torch.tensor(tracks_val[i][t_hist:]).float()
            delta = (y[:,[0,1,2,3]]-sols[2][:,[0,1,2,3]])*scaling
            lossi = torch.mean((torch.abs(delta)[eval_points]))*2
            val_loss = val_loss+lossi
            print("  val loss", i, lossi.item())
        val_loss = val_loss / len(tracks_val)    

        test_loss = 0
        for i in range(len(tracks_test)):
            model.zero_grad()
            sols = model(None,test_games[i],torch.tensor(tracks_test[i][:t_hist,:]).float(),test_conds[i][0],test_conds[i][1], history_size = pred_size)

            y = torch.tensor(tracks_test[i][t_hist:]).float()
            delta = (y[:,[0,1,2,3]]-sols[2][:,[0,1,2,3]])*scaling
            lossi = torch.mean((torch.abs(delta)[eval_points]))*2
            test_loss = test_loss+lossi
            print("  test loss", i, lossi.item())
        test_loss = test_loss / len(tracks_test)

        return val_loss, test_loss    
    
    
    # load best model
    best_loss = 1e9
    if False:
        model.load_state_dict(torch.load("trained_models/%s_game_model_vdelta%d_spacing%d_scaleaccel%d_vel_predsize%d_fold%d.torch" % (dataset_name,use_deltav,spacing,scale_accel,pred_size,fold,)))
        val_loss, test_loss = val_test_loss()
        print("Loaded Result", val_loss.item(), test_loss.item())
        best_loss = val_loss.item()


    ## learn game solver and game parameters
    def closure():
        loss = 0
        model.train()
        train_exp = 1.05
        #inds = np.random.choice(15,5,replace=False)
        for i in range(len(tracks_train)):
            model.zero_grad()
            sols = model(None,train_games[i],torch.tensor(tracks_train[i][:t_hist,:]).float(),train_conds[i][0],train_conds[i][1], history_size = pred_size)

            y = torch.tensor(tracks_train[i][t_hist:]).float()
            delta = (y[:,[0,1,2,3]]-sols[2][:,[0,1,2,3]])*scaling
            lossi = torch.mean((torch.pow(torch.abs(delta)[eval_points],train_exp)))*2
            loss = loss+lossi
            print("  train loss", i, lossi.item()**(1/train_exp)) 
        loss = loss / (len(tracks_train))

        val_loss = 0
        for i in range(len(tracks_val)):
            model.zero_grad()
            sols = model(None,val_games[i],torch.tensor(tracks_val[i][:t_hist,:]).float(),val_conds[i][0],val_conds[i][1], history_size = pred_size)

            y = torch.tensor(tracks_val[i][t_hist:]).float()
            delta = (y[:,[0,1,2,3]]-sols[2][:,[0,1,2,3]])*scaling
            lossi = torch.mean((torch.pow(torch.abs(delta)[eval_points],train_exp)))*2
            val_loss = val_loss+lossi
            print("  val loss", i, lossi.item())
        val_loss = val_loss / len(tracks_val)    

        loss = loss + 0.2*val_loss

        if loss.requires_grad:
            lbfgs_g.zero_grad()
            loss.backward()
        print("total train loss", loss.item()**(1/train_exp))
        return loss


    if True:
        for t in range(25):
            lbfgs_g.step(closure)    
            if t % 1 == 0:
                val_loss, test_loss = val_test_loss()
                print("Iteration", t, best_loss, val_loss.item(), test_loss.item())
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    torch.save(model.state_dict(), "trained_models/%s_game_model_vdelta%d_spacing%d_scaleaccel%d_vel_predsize%d_fold%d.torch" % (dataset_name,use_deltav,spacing,scale_accel,pred_size,fold,))

    # load best model
    model.load_state_dict(torch.load("trained_models/%s_game_model_vdelta%d_spacing%d_scaleaccel%d_vel_predsize%d_fold%d.torch" % (dataset_name,use_deltav,spacing,scale_accel,pred_size,fold,)))

    val_loss, test_loss = val_test_loss()
    print("Final  Result spacing%d pred_size%d dataset %s fold%d" % (spacing,pred_size, dataset_name, fold,), val_loss.item(), test_loss.item())


    
multi_pool = Pool(processes=5)
print("Running training on dataset %s ..." % (dataset_name,))
predictions = multi_pool.map(train_fold,list(range(4)))
multi_pool.close() 
multi_pool.join()
