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
This file contains the potential game class (PotGame) and some helper functions/classes
"""

import numpy as np
import torch
from torch.autograd import Variable
import itertools
import functools
from . import sdlbfgs



class PotGame(object):
    """ Game with potentials"""
    
    def __init__(self, convex_ndims, player_utilities):
        """
        Parameters
        ----------
          convex_ndims: int
               the number of variables of the game
          player_utilities: list of functions
               a list of functions each of which returns the utility of the respective agent
               each function gets called in the solve method, an must be able to handle the 
               arguments theta, sv,x, other_params = None.
        """
        self.player_utilities = player_utilities
        self.convex_ndims = convex_ndims
        self.num_players = len(player_utilities)
    
    def loss_fun(self, sv, x, theta,other_params):
        loss = 0
        for p in range(self.num_players):
            loss = loss - self.player_utilities[p](theta, sv,x,other_params)
        return loss
     
    def solve(self, theta, sv,other_params=None, history_size = 65):
        """
        solves the potential game with the given parameters and returns the subspace variables, the loss
        and the solution of the game x

        Parameters
        ----------
          theta: torch tensor
               the game parameters (as output by a preference revalation net
          sv: any
               the subspacevariables. Can be anythign, the player_utility functions must be able to handle it.
          other_params: any
               the player_utility functions get this as an additional arguments
          history_size:
               the size of the LBFGS solvers history (rank of the hessian approximation)

        """
        x, loss = PotGameSolverFct.apply(theta,sv,self,other_params, history_size)
        
        return sv, loss, x



def jacobian(y, x, create_graph=False):   
    """ helper method that computes the jacobian of y with respect to x"""
    jac = [ list() for i in range(len(x))]                                                                                        
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=True)
        for j, grad in enumerate(grad_x):
            jac[j].append(grad_x[j].reshape(x[j].shape))                                                           
        grad_y[i] = 0.                                                                                
    return [torch.stack(jac[i]).reshape(y.shape + x[i].shape) for i in range(len(x))]


class PotGameSolverFct(torch.autograd.Function):  
    """ Helper class that computes the gradient of the game with respect to
        the game parameters theta using the implicit function theorem
    """                    
    @staticmethod                                               
    def forward(ctx, theta, sv, game, other_params=None, history_size = 65):     
        with torch.enable_grad():   
            lr = 1.0
            #x, loss = game._solve_for_fct(theta, sv)
            x = Variable(torch.zeros((game.convex_ndims,)))
            x.requires_grad = True
            lbfgs = [torch.optim.LBFGS([x], lr = lr,  history_size=history_size, max_iter=15,tolerance_change=1e-05)]
            #lbfgs = [tg.sdlbfgs.SdLBFGS([x], lr = 0.7, history_size=history_size,lr_decay=False,max_iter = 10)]

            
            def closure():
                loss = game.loss_fun(sv, x, theta,other_params)
                if loss.requires_grad:
                    lbfgs[0].zero_grad()
                    loss.backward(retain_graph=True)
                return loss
            
            
            max_iter = 1000
            eps = 1e-3 # 1e-4 for train
            
            i = 0
            delta = 1e9
            xold = x.clone().detach()+1e1
            best_x = x.clone().detach()
            best_loss = torch.tensor([1e9])[0]
            nans = 0
            while i <  max_iter and delta > eps and nans < 5:
                assert not torch.isnan(best_x).any()
                assert not torch.isinf(best_x).any()                      
                lbfgs[0].step(closure)
                i += 1
                loss = closure()

                    
                if i % 50 == 0:
                    print(i,loss.item())
                
                #restart lbfgs if solution becomes NaN (see https://github.com/pytorch/pytorch/issues/5953)
                if torch.isnan(x).any() or torch.isinf(x).any() or torch.isinf(loss) or torch.isnan(loss):
                    print("LBFGS: NaN", i)
                    assert not torch.isnan(best_x).any()
                    assert not torch.isinf(best_x).any()
                    with torch.no_grad():
                        x[:] = best_x
                    lr = lr*0.5
                    lbfgs[0] = torch.optim.LBFGS([x], lr = lr,   history_size=history_size, max_iter=int(0.5*history_size))
                    xold =best_x+1e1
                    delta = torch.max(torch.abs(x-xold))                    
                    nans += 1
                else:
                    if loss < best_loss:
                        best_x = x.clone().detach()
                        best_loss = loss              
                    delta = torch.max(torch.abs(x-xold))                                        
                    xold = x.clone().detach()
                    
            x = best_x
        ctx.save_for_backward(theta, x)     
        ctx.sv = sv
        ctx.game = game
        ctx.other_params = other_params
        return x, best_loss                                       
                                                                
    @staticmethod                             
    def backward(ctx, dx, dloss):                                     
        """
        \subsubsection{The gradient of the game module}


        Let $g(\theta, a) := \nabla_{a} \potf (\theta, a)$. \todo{check if derivative w.r.t. $a$ makes sense}

        Let $h$ be the local mapping from params $\Theta$ to solution $A$, i.e., solving $g(\theta, h(\theta)) = 0$.
        Let $D g (\theta, a) = [J_1 (\theta, a), J_2 (\theta, a)]$, with $J_1$ corresponding to $\theta$ and $J_2$ to $a$.


        \begin{prop}
        Then, under some conditions,
        \begin{align*}
        \nabla h (\theta) = - [J_2(\theta, h(\theta))]^{-1}  J_1(\theta, h(\theta))
        \end{align*}
        \end{prop}        
        """
        theta, x = ctx.saved_variables
        sv = ctx.sv
        game = ctx.game
        other_params = ctx.other_params
        
        with torch.enable_grad(): 
            theta = theta.clone().detach()
            
            x = x.clone().detach()
            theta.requires_grad = True
            x.requires_grad = True
            
            pot = game.loss_fun(sv, x, theta,other_params)
            g = torch.autograd.grad(pot,x, create_graph=True, retain_graph=True)[0]
            
            J2,J1 = jacobian(g, [x,theta], create_graph=False)      
            
            J2 = J2 + torch.diag(torch.ones(J2.shape[0])*1e-5)
            X, LU = torch.solve(J1, J2)
            nabla_h = -X.transpose(0,1)
            
            assert not torch.isnan(nabla_h).any()
            assert not torch.isinf(nabla_h).any()

        return torch.mm(nabla_h,dx[:,None])[:,0], None, None, None, None

        
