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
This script trains the subspace predictor of the tgl/tgl-d method. The trained subspace predictor is stored in
the "trained_models" directory with a filename starting with the dataset name.
"""

import numpy as np
import torch
from tg.potential_game import PotGame

from tg.merging_game import pot_otherv, pot_mergerv, decode_sols, plot_game, GamePredictor,SubspacePredictor
from tg.merging_game import encode_sols2 as encode_sols

import tg.merging_game

import numpy as np
from data import hee_preprocess2

t_hist = 5
dataset_name = "hee-mytest"
spacing = 5


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
elif dataset_name.startswith("hee"):
    merge_tracks, scaling_fac = hee_preprocess2.get_all_merge_tracks2(skip=5, return_scaling=True)
    tracks = np.array([x for idx,x in enumerate(merge_tracks)])
    tracks = np.array([x[:8*5+1,:] for x in tracks])
    pred_length = 7*5+1 
    eval_points = (np.array([1,2,3,4,5,6,7])*25)//5
    scaling = torch.tensor(scaling_fac)
else:
    print("Unknown dataset", dataset_name)
    raise Exception()
    
from sklearn.model_selection import KFold
kf = KFold(n_splits=4, random_state=0, shuffle=True)

fold = 0

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


    theta0 = torch.log(torch.tensor([desired_v,desired_v,1,300, 0.0001,0.001,0.01]))
    theta = theta0.clone() #torch.log(torch.tensor([.3,0.8, 0.1,2,0.5]))-1
    theta.requires_grad = True


    ssp = SubspacePredictor(t_hist=t_hist,nhidden=64,layers=2)

    lbfgs_ssp = torch.optim.Adam(ssp.parameters(), lr = 0.005)


    # load best model
    try:
        #ssp.load_state_dict(torch.load("trained_models/highD_ssp_fold%d.torch" % (fold,)))    
        pass
    except:
        pass

    celoss = torch.nn.CrossEntropyLoss()
    nd = torch.distributions.normal.Normal(0,1)

    def loss_ss(gt, pred):
                before_after_gt = gt[0]
                t_gt = gt[1]

                lossi = 0
                lossce = celoss(pred[0][None,:],torch.tensor([before_after_gt]))*(t_gt<70)
                #lossi = lossi - nd.log_prob((t_gt-pred[1])/pred[2])*pred[2]
                losssq = ((t_gt-pred[1])).abs()**1.1
                return lossce, losssq


    def get_ss_val_test_loss(verbose=False):
                cetest_loss = 0            
                sqtest_loss = 0
                for i in range(len(tracks_test)):     
                    subspace = ssp(torch.tensor(tracks_test[i][:t_hist,:]).float())
                    celossi, sqlossi = loss_ss(test_conds[i][0], subspace)
                    if verbose:
                        print("Testtrack", i, test_conds[i][0],subspace)
                    #print(celossi)
                    cetest_loss = cetest_loss+celossi
                    sqtest_loss = sqtest_loss+sqlossi
                ceval_loss = 0            
                sqval_loss = 0            
                for i in range(len(tracks_val)):     
                    subspace = ssp(torch.tensor(tracks_val[i][:t_hist,:]).float())
                    celossi, sqlossi = loss_ss(val_conds[i][0], subspace)
                    if verbose:
                        print("Valtrack", i, val_conds[i][0],subspace)
                    ceval_loss = ceval_loss+celossi    
                    sqval_loss = sqval_loss+sqlossi    
                ceval_loss = ceval_loss / len(tracks_val)
                cetest_loss = cetest_loss / len(tracks_test)
                sqval_loss = sqval_loss / len(tracks_val)
                sqtest_loss = sqtest_loss / len(tracks_test)

                return ceval_loss, sqval_loss, cetest_loss, sqtest_loss


    if True:
        ## learn subsapce predictor

        def closure():
            ssp.train()
            loss = 0
            lossv = 0
            idx = np.random.choice(len(tracks_train), len(tracks_train), replace = False)            
            for i in range(len(idx)):     
                subspace = ssp(torch.tensor(tracks_train[idx[i]][:t_hist,:]).float())
                #print(subspace)
                lossi,_ = loss_ss(train_conds[idx[i]][0], subspace)
                loss = loss+lossi
            for i in range(len(tracks_val)):     
                subspace = ssp(torch.tensor(tracks_val[i][:t_hist,:]).float())
                #print(subspace)
                lossi,_ = loss_ss(train_conds[i][0], subspace)
                lossv = lossv+lossi
            loss = loss/len(idx)+lossv*0.1/len(tracks_val)
            if loss.requires_grad:
                lbfgs_ssp.zero_grad()
                loss.backward()    
                #torch.nn.utils.clip_grad.clip_grad_value_(ssp.parameters(),5)
            return loss

        best_loss = 1e9

        for t in range(200):
            lbfgs_ssp.step(closure)    
            if t % 1 == 0:
                loss = closure()   
                #print("Iteration", t, loss.item())
                ssp.eval()
                ceval_loss,sqval_loss, cetest_loss, sqtest_loss = get_ss_val_test_loss()
                #print("Val_loss, test_loss", ceval_loss, cetest_loss)
                if ceval_loss.item() < best_loss and t > 5:
                    best_loss = ceval_loss.item()
                    print("Iteration", t, "BEST Val_loss, test_loss", ceval_loss, cetest_loss)
                    torch.save(ssp.state_dict(), "trained_models/%s_ssp_spacing%d_fold%d.torch" % (dataset_name,spacing,fold,))

    # load best model
    ssp.load_state_dict(torch.load("trained_models/%s_ssp_spacing%d_fold%d.torch" % (dataset_name,spacing,fold,)))                   

    if True:
        ## learn subsapce predictor

        ## learn game solver and game parameters
        def closure():
            ssp.train()
            loss = 0
            idx = np.random.choice(len(tracks_train), len(tracks_train), replace = False)
            for i in range(len(idx)):     
                subspace = ssp(torch.tensor(tracks_train[idx[i]][:t_hist,:]).float())
                #print(subspace)
                _,lossi = loss_ss(train_conds[idx[i]][0], subspace)
                loss = loss+lossi
            lossv = 0
            for i in range(len(tracks_val)):     
                subspace = ssp(torch.tensor(tracks_val[i][:t_hist,:]).float())
                #print(subspace)
                _,lossi= loss_ss(train_conds[i][0], subspace)
                lossv = lossv+lossi
            loss = loss/len(idx)+lossv*0.1/len(tracks_val)
            if loss.requires_grad:
                lbfgs_ssp.zero_grad()
                loss.backward()    
                #torch.nn.utils.clip_grad.clip_grad_value_(ssp.parameters(),5)
            return loss/len(idx)

        best_loss = 1e9

        for t in range(1000):
            lbfgs_ssp.step(closure)    
            if t % 1 == 0:
                loss = closure()   
                #print("Iteration", t, loss.item())
                ssp.eval()
                ceval_loss,sqval_loss, cetest_loss, sqtest_loss = get_ss_val_test_loss()
                if sqval_loss.item() < best_loss:
                    best_loss = sqval_loss.item()
                    print("Iteration", t, "Val_loss, test_loss", sqval_loss, sqtest_loss)
                    torch.save(ssp.state_dict(), "trained_models/%s_ssp_spacing%d_fold%d.torch" % (dataset_name,spacing,fold,))

    # load best model
    ssp.load_state_dict(torch.load("trained_models/%s_ssp_spacing%d_fold%d.torch" % (dataset_name,spacing,fold,)))     

    ceval_loss,sqval_loss, cetest_loss, sqtest_loss = get_ss_val_test_loss(verbose=True)            
    print("Final dataset %s Fold" % (dataset_name,) , fold, "SSP val_loss, test_loss", ceval_loss, sqval_loss, cetest_loss, sqtest_loss)

