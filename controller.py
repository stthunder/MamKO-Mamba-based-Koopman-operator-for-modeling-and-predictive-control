import numpy as np
from cvxpy import *
import mosek
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
import time
import torch
from einops import rearrange, repeat, einsum
import casadi as ca



class Upper_MPC_mamba(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self._build_matrices(args)
        self.init = False
        self.next = False

        self.R = R
        self.Q = Q
        self.P = P


        self.data_save_x = []
        self.data_save_u = []

    def _get_set_point_u(self):
        pass



    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_dim']
        
        self.A_holder = Parameter((self.L, self.x_dim))              
        self.B_holder = Parameter((self.x_dim*self.L, self.u_dim+self.p_dim))  
        self.C_holder = Parameter((self.x_dim*self.L, self.x_use)) 


    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        self.create_prob(self.args)
        self._get_set_point_u()

    
    
    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.control_horizon+1))

        self.x_init = Parameter(self.x_dim)           

        
        objective = 0.
        constraints =  [self.x[:, 0] == self.x_init]


        for k in range(self.control_horizon-1):
            # k_u = k if k <= self.control_horizon-1 else self.control_horizon-1
            constraints += [self.x[:, k + 1] == multiply(self.A_holder[k,:],self.x[:, k]) + self.B_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ self.u[:,k]]
            j_ = self.C_holder[k*self.x_dim:(k+1)*self.x_dim,:].T @ self.x[:,k]
            objective += quad_form(j_-self.reference, self.Q)

            if k <= self.control_horizon-1 and k>0:
                objective += quad_form(self.u[:,k]-self.u[:,k-1],self.R)

            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        
        k = k+1
        j_ = self.C_holder[k*self.x_dim:(k+1)*self.x_dim,:].T @ self.x[:,k]
        objective += quad_form(j_-self.reference, self.P)
        self.prob = Problem(Minimize(objective), constraints)
        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """

        x_0 = (x_0-self.shift)/self.scale

        if not self.init:
            self.init = True
            self.start_x = x_0
            
            for i in range(self.L):
                self.data_save_x.append(self.start_x)
                self.data_save_u.append((np.zeros([self.u_dim])-self.shift_u)/self.scale_u)


        phi_0 = self.model.net.project_x(torch.from_numpy(x_0).to(torch.float32)) # project_x 


        add_x = torch.from_numpy(np.array(self.data_save_x).squeeze()).to(torch.float32)
        add_u = torch.from_numpy(np.array(self.data_save_u).squeeze()).to(torch.float32)


        delta, B, C = self.model.net.generate_matrices(add_x.unsqueeze(0), add_u.unsqueeze(0), phi_0.unsqueeze(0))
        A,B,C = self.model.net.discretion_matrices(self.model.net.A, B, C, delta)

        self.A_holder.value = A.squeeze().reshape(-1,self.x_dim).detach().numpy()
        self.B_holder.value = B.squeeze().reshape(-1,self.u_dim).detach().numpy()
        self.C_holder.value = C.squeeze().reshape(-1,self.x_use).detach().numpy()
        
        self.x_init.value = phi_0.detach().numpy()

        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})
        u = self.u[:, 0].value
        print(self.prob.objective.value)
  
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            su = u
            self.u_save = u
            u = u * self.scale_u + self.shift_u


        else:
            print("Error: Cannot solve mpc..")
            su = u
            u = u * self.scale_u + self.shift_u

        #--------------------store the old data------------------------#
        self.data_save_x.append(x_0)
        self.data_save_u.append(su)


        # delte some old data
        self.data_save_x.pop(0)
        self.data_save_u.pop(0)

        return u
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass



    

class Upper_MPC_DKO(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self._build_matrices(args)
        self.init = False
        self.next = False

        self.Q = Q
        self.R = R
        self.P = P



    def _get_set_point_u(self):
        pass

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_dim']



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        self.A_holder = self.model.net.A.detach().numpy()
        self.B_holder = self.model.net.B.detach().numpy()
        self.C_holder = self.model.net.C.detach().numpy()

        self.create_prob(self.args)
        self._get_set_point_u()

    
    
    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.control_horizon+1))

        self.x_init = Parameter(self.x_dim)        

        
        objective = 0.
        constraints =  [self.x[:, 0] == self.x_init]


        for k in range(self.control_horizon-1):

            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @ self.u[:,k]]
            j_ = self.C_holder @ self.x[:,k]
            objective += quad_form(j_-self.reference, self.Q)

            if k <= self.control_horizon-1 and k>0:
                objective += quad_form(self.u[:,k]-self.u[:,k-1],self.R)

            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        
        j_ = self.C_holder @ self.x[:,k+1]
        objective += quad_form(j_-self.reference, self.P)
        self.prob = Problem(Minimize(objective), constraints)
        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """

        x_0 = (x_0-self.shift)/self.scale

        phi_0 = self.model.net.project_x(torch.from_numpy(x_0).to(torch.float32)) # project_x 
        self.x_init.value = phi_0.detach().numpy()

        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})
        u = self.u[:, 0].value
        print(self.prob.objective.value)

        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            self.u_save = u
            u = u * self.scale_u + self.shift_u
            

        else:
            print("Error: Cannot solve mpc..")
            u = u * self.scale_u + self.shift_u

        return u
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass

class Upper_MPC_MLP(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model

        self.model.net.eval()

        self._build_matrices(args)
        self.init = False
        self.next = False

        self.Q = Q
        self.R = R
        self.P = P

        self.state_dict = self.model.net.state_dict()

        self.w = []

        for param_tensor in self.state_dict:
            self.w.append(self.state_dict[param_tensor].numpy())


    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_dim']
    
    def Reward_state(self,Z):
        Err = Z-self.reference
        return ca.mtimes([Err.T,self.Q, Err])
    
    def Reward_state_T(self,Z):
        Err = Z-self.reference
        return ca.mtimes([Err.T,self.P, Err])
    
    def Reward_input(self,U_0,U_1):
        Err = U_0-U_1
        return  ca.mtimes([Err.T,self.R,Err])



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)
        self.create_prob(self.args)
        self._get_set_point_u()


    def system_dynamics(self,x, u):
        
        m = ca.vertcat(x,u)

        j = 0
        while(j<len(self.w)):
            m = ca.mtimes(self.w[j],m)
            j+=1
            m = m + self.w[j]
            j+=1
            if j<=len(self.w)-2:
                m = ca.fmax(0, m)
        return m
    
    
    def create_prob(self, args):
        

        x =  ca.MX.sym('x',self.s_dim,self.control_horizon+1)
        u =  ca.MX.sym('u',self.a_dim,self.control_horizon)

        self.x0    = ca.MX.sym('x0', self.s_dim)
        self.J   = 0

        constraints = []
        constraints.append(x[:, 0] - self.x0)
        for k in range(self.control_horizon-1):
            constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k]))
            self.J += (self.Reward_state(x[:,k])+self.Reward_input(u[:,k],u[:,k+1]))
        
        k+=1
        self.J += self.Reward_state(x[:,k])

        k+=1
        self.J += self.Reward_state_T(x[:,k])


        nlp = {'x': ca.vertcat(ca.reshape(u, -1, 1), ca.reshape(x,-1, 1)),  
               'f': self.J, 
               'g': ca.vertcat(*constraints),
               'p': ca.vertcat(self.x0)}
        
        opts = {'ipopt': {'print_level': 0}}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbx = np.concatenate([np.tile(self.a_bound_low,self.control_horizon),  np.full(self.s_dim*(self.control_horizon+1), -np.inf)])  # No constraints on x
        self.ubx = np.concatenate([np.tile(self.a_bound_high,self.control_horizon), np.full(self.s_dim*(self.control_horizon+1), np.inf)]) 

        self.lbg = 0
        self.ubg = 0

        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """
        phi_0 = (x_0-self.shift)/self.scale
        if not self.init:
            self.init = True
            self.x0_guess = np.concatenate([np.zeros((self.a_dim*self.control_horizon, 1)), np.tile(phi_0, (self.control_horizon+1, 1)).T.reshape(-1, 1)])
        sol = self.solver(x0=self.x0_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=phi_0)

        U_opt = np.array(ca.vertsplit(sol['x'])[:self.a_dim*(self.control_horizon)]).flatten()
        X_opt = np.array(ca.vertsplit(sol['x'])[self.a_dim*(self.control_horizon):]).flatten()
        Pred_U = U_opt.reshape([self.a_dim,-1])
        Pred_X = X_opt.reshape([self.s_dim,-1])

        self.x0_guess = np.array(ca.vertsplit(sol['x'])).flatten()
        u_out  = Pred_U[:,0] * self.scale_u + self.shift_u
        
        return u_out
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass
    

    




if __name__ == '__main__':
    pass