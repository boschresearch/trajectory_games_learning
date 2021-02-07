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
This file includes the potential functions of the agents for the mergin scenarios, the corresponding 
game predictor, subspace predictor and helper functions
"""

import numpy as np
import torch
from torch import nn

elu = torch.nn.ELU()

def modexp(x):
    """ elu(x)+1, converts argument into a positive number"""
    return elu(x)+1

def imodexp(x):
    """ inverse of modexp function"""
    return torch.where(x>=1, x-1,torch.log(x))

def smoothmax(a,b,alpha = 1):
    """ smooth maximum of a and b"""
    m = torch.max(a,b)
    return (a*torch.exp(alpha*(a-m))+b*torch.exp(alpha*(b-m)))/(torch.exp(alpha*(a-m))+torch.exp(alpha*(b-m)))

other_start_pos = torch.tensor([0.,0.])
other_start_v = torch.tensor([0.5,0.])

merger_start_pos = torch.tensor([0.,0.])
merger_start_v = torch.tensor([0.7,0.])


lane_merger_y = 0
lane_normal_y = 1


alpha_accel = 1
alpha_v = 0.3
alpha_middle = 0.5
alpha_dist = 1
alpha_end_lane = .5

    
    
def pot_otherv(theta, sv,x, other_params = None):
    """ 
    potential function of the vehicle on the highway
    
    Parameters
    ----------
        theta: torch tensor
            the parameters of the game
        sv: list
            the subspace variables sv[0] is the indicator of before/after, sv[1] is the time of merge (in steps)
        x: torch tensor
            the encoded future positions of the agents
        other_params: torch tensor
            other parameters of the game which are not learned but are additional info, like starting speed etc.
    """
                
    other_desired_v = modexp(theta[0:1])+1e-5
    merger_desired_v = modexp(theta[1:2])+1e-5
    alpha_accel = modexp(theta[2:3])+1e-5
    alpha_v = 1.#modexp(theta[3:4])+1e-5
    alpha_middle = modexp(theta[4:5])+1e-5
    alpha_dist = modexp(theta[5:6])+1e-5
    alpha_end_lane = modexp(theta[6:7])+1e-5
    alpha_accel_y = modexp(theta[8:9])+1e-5
    if True:
        other_start_pos = other_params[0].to(x.device)
        other_start_v = other_params[1].to(x.device)

        merger_start_pos = other_params[2].to(x.device)
        merger_start_v = other_params[3].to(x.device)
        

    
    xo = x.view((-1,4))
    T = xo.shape[0]
    
    ts = torch.arange(T).to(x.device)[:,None]
    
    x = decode_x2(xo, sv, other_params)
    
    pos_t = x[:,0:2]
    
    assert not torch.isnan(pos_t).any()
    pos_p_t = torch.cat([other_start_pos[None,:],pos_t[:-1,:]],axis=0)
    v_t = torch.cat([other_start_v[None,:],pos_t-pos_p_t])
    
    delta_other_x = (x[:,2]-x[:,0])**2

    utility_distance = torch.where(ts[:,0] >= sv[1], -alpha_dist*(1/(0.1+delta_other_x)), torch.zeros(T).to(x.device))
    
    utility_middle = -alpha_middle*5*((pos_t[:,1]-lane_normal_y)**2)
    
    utility_v = - alpha_v*(v_t[-1:,0]-other_desired_v)**2
    
    scale_accel = 5
    
    utility_accel = - alpha_accel*((scale_accel*(v_t[1:,0]-v_t[0:-1,0]))**2)  
    
    utility_accel_y = - alpha_accel_y*((10*((v_t[1:,1]-v_t[0:-1,1]))**2))  
    
    assert not torch.isnan(utility_accel).any()
    
    utility = utility_middle.mean()+utility_v.mean()+utility_distance.mean()+utility_accel.mean()+utility_accel_y.mean()
    assert not torch.isnan(utility)

    return utility


def pot_mergerv(theta, sv,x, other_params = None):
    """ 
    potential function of the vehicle trying to merge onto the highway
    
    Parameters
    ----------
        theta: torch tensor
            the parameters of the game
        sv: list
            the subspace variables sv[0] is the indicator of before/after, sv[1] is the time of merge (in steps)
        x: torch tensor
            the encoded future positions of the agents
        other_params: torch tensor
            other parameters of the game which are not learned but are additional info, like starting speed etc.
    """
    other_desired_v = modexp(theta[0:1])+1e-5
    merger_desired_v = modexp(theta[1:2])+1e-5
    alpha_accel = modexp(theta[2:3])+1e-5
    alpha_v = 1.#modexp(theta[3:4])+1e-5
    alpha_middle = modexp(theta[4:5])+1e-5
    alpha_dist = modexp(theta[5:6])+1e-5
    alpha_end_lane = modexp(theta[6:7])+1e-5
    end_merging_lane_x = theta[7:8]
    alpha_accel_y = modexp(theta[8:9])+1e-5
    alpha_vel_y = modexp(theta[9:10])+1e-5
    alpha_smoothmax = modexp(theta[10:11])*0.1+1e-5
    
    if True:
        other_start_pos = other_params[0].to(x.device)
        other_start_v = other_params[1].to(x.device)

        merger_start_pos = other_params[2].to(x.device)
        merger_start_v = other_params[3].to(x.device)        


    
    xo = x.view((-1,4))
    T = xo.shape[0]
    
    ts = torch.arange(T).to(x.device)[:,None]   

    
    x = decode_x2(xo, sv, other_params)
    
    pos_t = x[:,2:4]
    
    assert not torch.isnan(pos_t).any()
    pos_p_t = torch.cat([merger_start_pos[None,:],pos_t[:-1,:]],axis=0)
    v_t = torch.cat([merger_start_v[None,:],pos_t-pos_p_t])

    delta_other_x = (x[:,2]-x[:,0])**2
    
    utility_distance = torch.where(ts >= sv[1], -alpha_dist*(1/(0.1+delta_other_x)), torch.zeros((T,)).to(x.device))
    
    if sv[1] < T:
        utility_middle = -alpha_middle*5*((pos_t[T-4:,1:2]-lane_normal_y)**2)
    else:
        utility_middle = torch.zeros(1).to(x.device)
    
    utility_v = - alpha_v*(v_t[-1:,0]-merger_desired_v)**2
    scale_accel = 5
    utility_accel = - alpha_accel*((scale_accel*(v_t[1:,0]-v_t[0:-1,0]))**2)  
    utility_accel_y = - alpha_accel_y*((10*(v_t[1:,1]-1*v_t[0:-1,1]))**2)  
    utility_vel_y = - alpha_vel_y*((3*(smoothmax(v_t[:,1].abs()-0.03,v_t[:,1]*0,90)+0.03))**2)
    assert not torch.isnan(utility_accel).any()
    utility_end_of_lane = -alpha_end_lane*torch.where((ts[:,0] < sv[1])*(sv[1]<(pos_t.shape[0]-2)), smoothmax(alpha_smoothmax*(pos_t[:,0] - end_merging_lane_x),pos_t[:,0]*0),torch.zeros(T).to(x.device))
    utility = utility_middle.mean()+utility_v.mean()+utility_distance.mean()+utility_accel.mean()+utility_accel_y.mean()+utility_vel_y.mean()+utility_end_of_lane.mean()   
    assert not torch.isnan(utility)
    return utility



def decode_sols(sols,other_params):
    """ helper function that decodes the future positions of the agents from the encoded representation"""
    new_sols = []
    for sol in sols:

        c = sol[0]
        l = sol[1]
        x = sol[2]
        
        x = decode_x2(x, c, other_params)
        
        new_sols.append((c,l,x))
    return new_sols

class RandMult(nn.Module):
    """ helper neural network layer that is similar to dropout and multiplies the inputs with a random value close to 1.0"""
    def __init__(self, fac = 0.06):
        super().__init__()
        self.fac = fac

    def forward(self, x):
        if self.training:
            rand = torch.rand(x.shape).to(x.device)*self.fac*2
        else:
            rand = self.fac
        return x*(1-self.fac+rand)

class GamePredictor(nn.Module):
    """
    The module which predicts the future positions of the agents by invoking the potential game solver.
    """
    def __init__(self,theta_init, num_local_params, count_local_params, t_hist = 7, nhidden=16, nlayers = 2, velocity = 0, use_deltav = True, deltav_plus = 1.1, deltav_minus = 0.3):  
        """
        Parameters
        ----------

            theta_init: torch tensor
                the initial values of the game parameters
            num_local_params: unused
            count_local_params: unused
            t_hist: int
                the number of time steps considered as history
            nhidden: int
                the nummber of hidden neurons for the preference revalation net
            nlayers: int
                the number of hidden layers of the preference revalation net
            deltav: boolean
                wether the preference revelation should predict an absolute velocity or a delta-velocity
            deltav_plus: float
                the maximum increase of the predicted v vs. old v
            deltav_minus: float
                the maximum decrease of the predicted v vs. old v
        """
        super().__init__()
        if nlayers == 2:
            self.lstm = nn.Sequential(nn.Linear(4*(t_hist-velocity)+velocity,int(1*nhidden)),nn.CELU(),nn.Linear(int(1*nhidden),int(1.5*nhidden)),nn.CELU(), RandMult())
        else:
            self.lstm = nn.Sequential(nn.Linear(4*(t_hist-velocity)+velocity,int(1.5*nhidden)),nn.CELU(),RandMult())
            
        self.out_encoder = nn.Linear(int(1.5*nhidden),2)
        self.t_hist = t_hist
        self.theta = nn.Parameter(theta_init,requires_grad=True)
        self.velocity = velocity
        self.use_deltav = use_deltav
        self.deltav_plus = deltav_plus
        self.deltav_minus = deltav_minus

    def forward(self, local_param, game, x, sv, other_params=None, local_params = None, verbose=False, validate = False, history_size = 65):
        """
        The method which predicts the future trajectories of the agents

        Parameters
        ----------
            local_param: unused
            game: PotentialGame instance
                the games which should be solved
            x: torch tensor
                the previous trajectory of the agents
            other_params: torch tensor
                the other parameters of the game
            local_params: unused
            verbose: boolean
                if True print some additional output
            validate: boolean
                if True do some additional postprocessing of the preference revalation net output, should be disabled
            history_size: int
                the history size (rank) of the  LBFGS hessian approximation for the game solve
                
        """
        
        theta = self.theta

        xh = x = x[:self.t_hist]
        vhistory = xh[-1,[0,2]] - xh[-2,[0,2]]

        if self.use_deltav:
             if self.velocity == 0:
                 x = torch.cat([x[:,0:2]-x[self.t_hist-1:self.t_hist,2:4],x[:,2:4]-x[self.t_hist-1:self.t_hist,2:4]],dim=1)
                 ho = self.lstm(x[:self.t_hist].view(1,-1))
             else:
                 v = torch.cat([x[1:,0:2]-x[:-1,0:2],x[1:,2:4]-x[:-1,2:4]],dim=1)
                 ho = self.lstm(torch.cat([v.view(-1),x[-1:,0]-x[-1:,2]],dim=0).view(1,-1))
                 
             if local_param is None:
                 deltav = self.out_encoder(ho[0:1])[0]*0.05
                 deltav = torch.sigmoid(deltav)*(self.deltav_plus+self.deltav_minus)-self.deltav_minus # limit to sensful deltav range
                 thetav = imodexp(vhistory+deltav)
        else:
             if self.velocity == 0:
                 x = torch.cat([x[:,0:2]-x[self.t_hist-1:self.t_hist,2:4],x[:,2:4]-x[self.t_hist-1:self.t_hist,2:4]],dim=1)
                 ho = self.lstm(x[:self.t_hist].view(1,-1))
             else:
                 v = torch.cat([x[1:,0:2]-x[:-1,0:2],x[1:,2:4]-x[:-1,2:4]],dim=1)
                 ho = self.lstm(torch.cat([v.view(-1),x[-1:,0]-x[-1:,2]],dim=0).view(1,-1))
                 
             if local_param is None:
                 thetav = self.out_encoder(ho[0:1])[0]*0.05
                 if verbose:
                     print(" desired velocity", modexp(thetav).cpu().detach().numpy())
                     print(" desired before/after", sv[0])
                 if validate:
                     if True: # validate
                         sv = list(sv)
                         desv = modexp(thetav)
                         oldv = xh[self.t_hist-1,[0,2]] - xh[self.t_hist-2,[0,2]]
                         oldx = xh[self.t_hist-1,[0,2]]
                         avgv = (0.1*desv+0.9*oldv)
                         newx = oldx+25*avgv
                         deltax = newx[0] - newx[1]
                         #if deltax > 0:
                         #     sv[0] = 1
                         #if deltax < -0:
                         #     sv[0] = 0
                         #if verbose:
                         #     print(newx, " valdidated before/after", sv[0])
                         
                         if sv[0] == 1: # merge behind
                             newvo = torch.clamp(desv[0:1],0.96*oldv[0],1.04*oldv[0])
                             newvm = torch.clamp(desv[1:2],1.0*oldv[1],1.25*oldv[1])
                         elif sv[0] == 0: # merge before
                             newvm = torch.clamp(desv[1:2],min(1.25*oldv[1],0.75*oldv[0]),1.25*oldv[1])
                             if newvm[0] > oldv[0]:
                                 newvo = torch.clamp(desv[0:1],0.96*oldv[0],1.04*oldv[0])
                             else:
                                 newvo = torch.clamp(desv[0:1],0.75*oldv[0],1.0*oldv[0])

                         else:
                             assert 1==2

                     thetav = imodexp(torch.cat([newvo,newvm]))    
                     if verbose:
                         print(" valdidated desired velocity", sv[0], modexp(thetav).cpu().detach().numpy())
                
            
        theta = torch.cat([thetav,theta],axis=0)
        #print("other params", other_params)
        import time
            
        t1 = time.time()
                    
        sol = list(game.solve(theta,sv,other_params,history_size=history_size))
            
        t2 = time.time()
            
        #print(" Solution took", t2-t1)
            
        sol[2] = sol[2].view(-1,4)
        sols = decode_sols([sol],other_params)[0]
        
        return sols


class SubspacePredictor(nn.Module):
    """ subspace predictor module which predicts the subspace from the past agents trajectory"""

    def __init__(self,t_hist=10,nhidden=64, layers = 1):
        """
        Parameters
        ----------

            t_hist: int
                number of time steps to consider when predicting the subspace
            layers: int
                the number of hidden layers of the subspace predictor
        """
        super().__init__()
        self.nhidden = nhidden
        dr = 0.6
        self.lstm1 = nn.Sequential(nn.Linear(4*t_hist+1,nhidden//4),nn.ELU(),nn.Dropout(dr),nn.Linear(nhidden//4,nhidden//8),nn.ELU(),nn.Dropout(dr))#,nn.Linear(nhidden,nhidden),nn.LeakyReLU())
        self.lstm2 = nn.Sequential(nn.Linear(4*t_hist+1,nhidden//1),nn.LeakyReLU(),nn.Dropout(dr),nn.Linear(nhidden//1,nhidden//2),nn.LeakyReLU())
        self.out_encoder_sv0 = nn.Linear(nhidden//8,2)
        self.out_encoder_gauss = nn.Linear(nhidden//2,2)

        self.t_hist = t_hist
        
    def forward(self, x):
        """
        Predicts the subspace from the past agents trajectory

        Parameters
        ----------
            x: torch tensor
                the past agents trajectories

        Returns sv[0], sv[1], sigma
            sv[0]: int
                the before/after flag
            sv[1]: float
                the time point of the merge in steps 
            sigma:
                the estimated standard deviation of the time point of the merge
        """
        #input = torch.cat([x[:,0:2],x[:,2:4]],dim=1)
        input = x = x[:self.t_hist,:]
        input = torch.cat([x[:,0:1]-x[-1:,2:3],x[:,1:2],x[:,2:3]-x[-1:,2:3],x[:,3:4]],dim=1)
        input = torch.cat([input.view(-1),x[-1:,2:3].view(-1)])
        ho1 = self.lstm1(input.view(1,-1))#input[None,:,:])
        ho2 = self.lstm2(input.view(1,-1))#input[None,:,:])
        sv0 = self.out_encoder_sv0(ho1[0:1,:])[0]
        gaussp = self.out_encoder_gauss(ho2[0:1,:])[0]*0.1
        

        return sv0, modexp(gaussp[0]), modexp(gaussp[1])


def encode_sols2(trajs,offset = 5):
    """ helper function to encode the trajectories into a delta representation which does not require
        constraints in the solver"""
    new_sols = []
    for traj in trajs:
        x = torch.tensor(traj)
        c = [0,0]
        
        first_ind = (x[offset:,3] >= 0.5).nonzero()
        if len(first_ind)>0:
            first_ind = first_ind[0][0]
        else:
            first_ind = x.shape[0]-1-offset
            
        c[1] = first_ind # merge at timestep c[1]
        
        if x[first_ind+1,2] >= x[first_ind+1,0]:
            c[0] = 0 # merger in front
        else:
            c[0] = 1 # other in front
        
        other_params = []
        
        v_other = (x[offset-1,:2]-x[0,:2]).float()/(offset-1)
        v_merger = (x[offset-1,2:4]-x[0,2:4]).float()/(offset-1)
        x_other = x[offset-1,:2]
        x_merger = x[offset-1,2:]

        other_params.append(x_other.float())
        other_params.append(v_other)
        other_params.append(x_merger.float())
        other_params.append(v_merger)
        other_params.append(3.7)
            
        new_sols.append([c,other_params])
    return new_sols


def _get_x_merger_other(sv,x,other_params):
    """ helper function to further decode the encoded representation"""

    if other_params is not None:
        other_start_pos = other_params[0]
        other_start_v = other_params[1]

        merger_start_pos = other_params[2]
        merger_start_v = other_params[3]
        end_merging_lane_x = other_params[4]
    
    ts = torch.arange(x.shape[0]).to(x.device)[:,None]
    vmean = (merger_start_v[0]+other_start_v[0])*0.5
    disth = 0.5*(other_start_pos[0]-merger_start_pos[0]).abs()
    xmean = 0.5*(merger_start_pos[0]+other_start_pos[0])+x[:,0:1]+(torch.arange(x.shape[0])+1).to(x.device).float()[:,None]*vmean
    if sv[0] == 1:
        xmerger = torch.where(ts >= sv[1], xmean - modexp(x[:,2:3]+disth-1), merger_start_pos[0]+torch.cumsum(modexp(x[:,2:3]+vmean-1),dim=0))
        xother = torch.where(ts >= sv[1], xmean + modexp(x[:,2:3]+disth-1), other_start_pos[0]+torch.cumsum(modexp(x[:,0:1]+vmean-1),dim=0)) 
    else:
        xmerger = torch.where(ts >= sv[1], xmean + modexp(x[:,2:3]+disth-1), merger_start_pos[0]+torch.cumsum(modexp(x[:,2:3]+vmean-1),dim=0))
        xother = torch.where(ts >= sv[1], xmean - modexp(x[:,2:3]+disth-1), other_start_pos[0]+torch.cumsum(modexp(x[:,0:1]+vmean-1),dim=0)) 
    return xmerger, xother

def decode_x2(x, c, other_params):
        if True:
            other_start_pos = other_params[0]

            other_start_v = other_params[1]

            merger_start_pos = other_params[2]
            merger_start_v = other_params[3]
            end_merging_lane_x = other_params[4]
        
        ts = torch.arange(x.shape[0]).to(x.device)[:,None]
        xm,xo = _get_x_merger_other(c,x,other_params)
        
        x = x.index_copy(1,torch.tensor([2]).to(x.device), xm)
        x = x.index_copy(1,torch.tensor([1]).to(x.device),x[:,1:2] + lane_normal_y)
        x = x.index_copy(1,torch.tensor([0]).to(x.device), xo)
        x = x.index_copy(1,torch.tensor([3]).to(x.device),torch.where(ts >= c[1], 0.5+modexp(0.1*x[:,3:4]-2),0.5-modexp(0.1*x[:,3:4]-2)))

        return x    

def save_trajectories(file_name,sol,gt,t_hist=5):
    import pickle
    
    obj = {}
    obj["history_other"] = gt[:t_hist,0:2]
    obj["history_merger"] = gt[:t_hist,2:4]
    obj["gt_other"] = gt[t_hist:,0:2]
    obj["gt_merger"] = gt[t_hist:,2:4]
    obj["predicted_other"] = sol[2].cpu().detach().numpy()[:,0:2]
    obj["predicted_merger"] = sol[2].cpu().detach().numpy()[:,2:4]
    
    pickle.dump( obj, open( file_name, "wb" ) )
    print("saved file", file_name)

def plot_game(sol,other_params, gt = None):
    other_start_pos = other_params[0]
    other_start_v = other_params[1]

    merger_start_pos = other_params[2]
    merger_start_v = other_params[3]
    end_merging_lane_x = other_params[4]
        
    import matplotlib
    from matplotlib import rc
    rc('animation', html='html5')    
    from matplotlib import pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML
    
    x = sol[2].detach().numpy()
    maxx = max(np.max(x[:,0]),np.max(x[:,2]))
    minx = min(np.min(x[:,0]),np.min(x[:,2]))
    if gt is not None:
        maxx = max(maxx,max(np.max(gt[:,0]),np.max(gt[:,2])))
        minx = min(minx, min(np.min(gt[:,0]),np.min(gt[:,2])))
        
    
    fig = plt.figure()
    ax = plt.axes(xlim=(minx-1, maxx), ylim=(-1, 2))
    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)
    if gt is not None:
        line3, = ax.plot([], [], lw=1)
        line4, = ax.plot([], [], lw=1)
        
    lane_marker, = ax.plot([minx-1,maxx], [lane_normal_y/2,lane_normal_y/2], lw=2)
    lane_end, = ax.plot([end_merging_lane_x,end_merging_lane_x], [-lane_normal_y/2,lane_normal_y/2], lw=2)
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        if gt is not None:
            line3.set_data([], [])
            line4.set_data([], [])
        return line1,line2,lane_marker,lane_end

    def animate(i):
        line1.set_data(x[:i,0], x[:i,1])
        line2.set_data(x[:i,2], x[:i,3])
        if gt is not None:
            line3.set_data(gt[:i,0], gt[:i,1])
            line4.set_data(gt[:i,2], gt[:i,3])
        return line1,line2,lane_marker,lane_end
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=x.shape[0], interval=500, blit=True)

    HTML(anim.to_html5_video())
    plt.show()
    
    
