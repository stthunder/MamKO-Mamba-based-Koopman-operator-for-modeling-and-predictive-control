import torch
import numpy as np
import math
import copy

import torch.nn as nn
import torch.optim as optim
import torch.distributions as torchd
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
import torch.nn.init as init


from scipy.stats import unitary_group
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


SCALE_DIAG_MIN_MAX = (-20, 2)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, mean=0,std=0.1)

class Koopman_Desko(object):
    """
    """
    def __init__(
        self,
        args,
        **kwargs
    ):
        self.shift = None
        self.scale = None
        self.shift_u = None
        self.scale_u = None

        self.loss = 0

        self.loss_store = 0
        self.loss_store_t = 0

        if args['control']:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        args['device'] = self.device
        self.net = Mamba(args)
        self.net.apply(weights_init)

        self.net_para = {}
        self.loss_buff = 100000

        self.stable_x = []
        self.stable_u = []

        self.optimizer1 = optim.Adam([{'params': self.net.parameters(),'lr':args['lr1'],'weight_decay':args['weight_decay']}])
        self.optimizer1_sch = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=args['optimize_step'], gamma=args['gamma']) 

        self.MODEL_SAVE = args['save_model_path']
        self.OPTI1 = args['save_opti_path']

        self.loss_t = 0

        self.net.to(self.device)



    def learn(self, e, x_train,x_val,x_test,args):

        #-----------------------------------------------validation------------------------------------------------------#
        self.loss = 0
        count = 0
        self.train_data = DataLoader(dataset = x_val, batch_size = args['batch_size'], shuffle = False, drop_last = False)
        with torch.no_grad():
            for x_,u_ in self.train_data:
                self.pred_forward(x_,u_,args)
                count += 1

        self.loss_store = self.loss.detach().cpu().numpy()/count

        #-----------------------------------------------test------------------------------------------------------------#
        self.loss = 0
        count = 0
        self.train_data = DataLoader(dataset = x_test, batch_size = args['batch_size'], shuffle = True, drop_last = False)
        with torch.no_grad():
            for x_,u_ in self.train_data:
                self.pred_forward(x_,u_,args)
                count += 1

        self.loss_store_t = self.loss.detach().cpu().numpy()/count

        #-----------------------------------------------train-----------------------------------------------------------#
        # TODO: NOT SHUFFLE
        self.train_data = DataLoader(dataset = x_train, batch_size = args['batch_size'], shuffle = True, drop_last = False)
        self.loss = 0
        count = 0
        loss_buff = 0

        for x_,u_ in self.train_data:
            self.loss = 0
            self.pred_forward(x_,u_,args)
            loss_buff += self.loss
            count += 1
            self.optimizer1.zero_grad()
            self.loss.backward()
            self.optimizer1.step()

        self.optimizer1_sch.step()
        loss_buff = loss_buff/count

        if loss_buff<self.loss_buff:
            self.net_para = copy.deepcopy(self.net.state_dict())
            self.loss_buff = loss_buff

        print('epoch {}: loss_traning data {} loss_val data {} test_data {} minimal loss {} learning_rate {}'.format(e,loss_buff,self.loss_store,self.loss_store_t,self.loss_buff,self.optimizer1_sch.get_last_lr()))

    def test_(self,test,args):
        self.test_data = DataLoader(dataset = test, batch_size = 10, shuffle = True)
        for x_,u_ in self.test_data:
            self.pred_forward_test(x_,u_,args)

    def pred_forward(self,x,u,args):
        old_horizon = args['old_horizon']
        loss,_ = self.net(x.to(self.device),u.to(self.device))
        self.loss += loss


    def pred_forward_test(self,x,u,test,args,e=0):
        x_pred_list = []
        x_real_list = []
        x_time_list = []
        plt.close()
        f, axs = plt.subplots(args['state_dim'], sharex=True, figsize=(15, 15))
        time_all = np.arange(x.shape[1])
        print("done")
        self.loss_t = 0
        count = 0

        if test:
            for i in range(args['old_horizon'],x.shape[1]-args['pred_horizon'],args['pred_horizon']-1):
                x_pred = x[:,i-args['old_horizon']:i+args['pred_horizon']]
                u_pred = u[:,i-args['old_horizon']:i+args['pred_horizon']-1]
                x_pred_list_buff,x_real_list_buff = \
                    self.pred_forward_test_buff(x_pred,u_pred,args)
                x_pred_list.append(x_pred_list_buff)
                x_real_list.append(x_real_list_buff)
                x_time_list.append(np.arange(i,i+args['pred_horizon']))
                count+=1
            print(self.loss_t/count)
            # to show the performance of modeling 
            for i in range(args['state_dim']):
                axs[i].plot(time_all, x[:, :,i].T, 'k')
                for j in range(len(x_time_list)):
                    axs[i].plot(x_time_list[j], x_pred_list[j][:,0,i], 'r')
            
            plt.xlabel('Time Step')
            plt.savefig('data/mamba/predictions_' + str(e) + '.png')
            print("plot")

            if e == args['num_epochs']-1:
                np.save('Prediction/mamba/x_pred.npy',np.array(x_pred_list))
                np.save('Prediction/mamba/x_time.npy',np.array(x_time_list))
                np.save('Prediction/mamba/x_all_time.npy',np.array(time_all))
                np.save('Prediction/mamba/x_.npy',np.array(x))
                print('store!!!!')

            return x_pred_list,x_real_list
        ##----------------------------------------------##
        else:
            return self.pred_forward_test_buff(x,u,args)
    
    def pred_forward_test_buff(self,x,u,args):
        
        pred_horizon = args['old_horizon']
        loss,result = self.net(x.to(self.device),u.to(self.device))
        self.loss_t += loss

        return result,x[:,pred_horizon:,:]



    def parameter_store(self,args):
        """
        TODO: store data from mamba block
        """
        torch.save(self.net_para,self.MODEL_SAVE)
        torch.save(self.optimizer1.state_dict(),self.OPTI1)

        print("store!!!")
    

        
        
    def parameter_restore(self,args):
        """
        restore data from mamba block
        """
        if args['control']:
            self.device = torch.device('cpu')

        self.net = Mamba(args)
        self.net.load_state_dict(torch.load(self.MODEL_SAVE,map_location=self.device))    
        self.net.to(self.device)

        self.optimizer1 = optim.Adam([{'params': self.net.parameters(),'lr':args['lr1'],'weight_decay':args['weight_decay']}])


        # self.net.eval()
        print("restore!")


    

class Mamba(nn.Module):
    """
    use some concept of Mamba, design this block
    """
    def __init__(self, args):
        """A mamba model for LTV system"""
        super().__init__()
        self.L = args['pred_horizon']
        self.O = args['old_horizon']
        self.N = args['latent_dim']
        self.S = args['state_dim']
        self.d_conv = args['d_conv']
        self.U = args['act_dim']
        self.delta = args['delta']

        self.device = args['device']
        #---------------------------------without initilization---------------------------#
        # self.A = torch.abs(torch.randn(self.N, 1)) # seems only on the diagnoal => connection between each state???]

        self.A = torch.randn(self.N, 1)/20# TODO: This initial seems better
        self.A = nn.Parameter(self.A) # Stable seems like
        # self.A_ = self.P_inv*torch.diag(self.A.squeeze())*self.P_
        #---------------------------------without initilization---------------------------#
        self.x_project  = nn.Linear(self.S+self.U,self.N) # x from original d to N dimension koopman
        # self.x_project  = nn.Linear(self.S,self.N)
        self.x_project_ = nn.Linear(self.O-1,self.L)
        self.x_generate        = nn.Linear(self.N+self.U,self.delta+self.N*(self.U)+self.N*self.S)
        self.x_generate_deleta = nn.Linear(self.N,self.delta)
        self.dt_project = nn.Linear(self.delta,self.N)
         
        self.conv1d_all = nn.Conv1d(
            in_channels=self.N+self.U,
            out_channels=self.N+self.U,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.N+self.U,
            padding=self.d_conv - 1, #TODO: test
        )# conv on the time sequence for states and input

        self.conv1d_states = nn.Conv1d(
        in_channels=self.N,
        out_channels=self.N,
        bias=True,
        kernel_size=self.d_conv,
        groups=self.N,
        padding=self.d_conv - 1,
        )# conv on the time sequence for all the states

        self.u_project   = nn.Linear(self.U,self.U)
        self.x_project_0 = nn.Linear(self.S,args['hidden_dim'])
        self.x_project_1 = nn.Linear(args['hidden_dim'],self.N)

        

        init.kaiming_uniform_(self.conv1d_states.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv1d_all.weight, mode='fan_in', nonlinearity='relu')

    def forward(self,x, u):
        """
        seq2seq structure for training
        old data x(0:9) u(0:9) p(0:9)             -> determine A B C delta for time 1:10      
        current data                              -> project to Z
        new data (for control) u(10:19) p(1:10)   -> project to g with linear layer and elu
        x(B,L,D_x) u(B,L*2,D_u) p(B,L*2,D)
        """
        u = u[:,:,:self.U] # TODO: FOR CONVENIENCE

        z_0   = self.project_x(x[:,self.O-1,:])
        delta, B, C = self.generate_matrices(x,u,z_0)

        deltaA,deltaB,C = self.discretion_matrices(self.A, B, C, delta)

        u_0 = self.project_u(u[:,self.O-1:,:])

        return self.selective_scan(z_0, u_0, deltaA, deltaB, C, x)

    def generate_matrices(self,x,u,z_0):
        """
        Generate delta B C via old data
        input:  old data x(0:9) u(0:9) p(0:9)
        output: delta B C
        """
        if u.dim() ==2:
            u = u.unsqueeze(2)

        x_all = self.project_x(x[:,:self.O-1,:])
        u_all = self.project_u(u[:,:self.O-1,:])
        old_data = torch.cat((x_all,u_all),dim=2)
        old_data = F.silu(old_data)
        old_data = rearrange(old_data, 'B L D -> B D L')
        old_data = self.conv1d_all(old_data)[:,:,-self.L:]
        old_data = rearrange(old_data, 'B D L -> B L D')
        # old_data = F.silu(old_data)
        old_data = F.silu(old_data)
        
        old_data = self.x_generate(old_data) #  B L D -> B L delta+2*N
        (delta, B, C) = old_data.split(split_size=[self.delta, self.N*(self.U), self.N*self.S], dim=-1)

        delta = F.softplus(self.dt_project(delta))# B L N
        self.delta_save = delta
        batch, length , _ = B.shape
        B =  B.reshape(batch, length , self.N, self.U)
        C =  C.reshape(batch, length , self.N, self.S)

        return delta, B, C



    def selective_scan(self,z0, u, deltaA, deltaB, C,x):
        """
        z(k+1) = deltaA(k) z(k) + deltaB(k) u(k)
        x(k)   = C z(k)
        L  generate L+1
        """
        ys = []    
        loss = nn.MSELoss()  
        loss_ = 0
        for i in range(self.L): # The same as in the controller
            z0 = torch.mul(z0,deltaA[:,i,:])+einsum(deltaB[:,i,:],u[:,i,:],'B N L, B L -> B N')
            y = einsum(C[:,i,:,:],z0,'B L N, B L -> B N')
            loss_+= loss(y,x[:,self.O+i,:])
            ys.append(y.detach().cpu().numpy())

        return loss_/self.L, np.array(ys)
            


    def discretion_matrices(self, A, B, C, delta):
        """
        Generate A B after discretion_matrices
        delta
        """
        # A = -torch.exp(A)
        A = -F.celu(A)
        # A = -F.silu(A)
        # A = -F.celu(A)
        # A = -F.elu(A)
        # A = -F.elu(A) #TODO unstable system
        # print(A)
        deltaA = torch.exp(einsum(delta, A.squeeze(), 'B L N, N -> B L N')) # delta: B L N
        deltaA_cum = einsum((deltaA-1),(1/A).squeeze(),'B L N, N->B L N')
        deltaB     = einsum(deltaA_cum, B, 'B L N, B L N C -> B L N C')
        return deltaA, deltaB, C
        

    def project_x(self, x_0):
        """
        Project the state x into high-dimensional space
        """
        # return F.silu(self.x_project_0(x_0))
        # return F.relu(self.x_project_0(x_0))
        # return self.x_project_1(F.silu(self.x_project_0(x_0)))
        return self.x_project_1(F.relu(self.x_project_0(x_0)))

    def project_u(self, u):
        """
        Project the input u into high-dimensional space
        """
        # return F.elu(self.u_project(u))
        return u


if __name__ == '__main__':
    pass    
