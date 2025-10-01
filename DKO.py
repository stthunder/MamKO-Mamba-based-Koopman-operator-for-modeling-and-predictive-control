import torch
import numpy as np
import math
import copy

import torch.nn as nn
import torch.optim as optim
import torch.distributions as torchd
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from torch import einsum
from einops import rearrange, repeat
import torch.nn.functional as F

from scipy.stats import unitary_group
#===========for matplotlib==============#
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
        self.net = DKO(args)
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
        self.train_data = DataLoader(dataset = x_val, batch_size = args['batch_size'], shuffle = True, drop_last = False)
        with torch.no_grad():
            for x_,u_ in self.train_data:
                self.pred_forward(x_,u_,args)
                count += 1
        # print(self.net.mex_A)
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
        # loss = nn.MSELoss()
        loss,_ = self.net(x.to(self.device),u.to(self.device))
        self.loss += loss
        # self.loss += loss(self.net(x.to(self.device),u.to(self.device)),x[:,old_horizon:,:].to(self.device))
        # print(torch.mean(self.net(x.to(self.device),u.to(self.device))-x[:,old_horizon:,:].to(self.device),dim=(0,2)))
    
    def pred_forward_test(self,x,u,test,args,e=0):
        x_pred_list = []
        x_real_list = []
        x_time_list = []
        # plt.close()
        plt.figure(1)
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
            plt.savefig('data/DKO/predictions_' + str(e) + '.png')
            print("plot")
            if e == args['num_epochs']-1:
                # plt.savefig('data/predictions_' + str(e) + '.pdf')
                np.save('Prediction/DKO/x_pred.npy',np.array(x_pred_list))
                np.save('Prediction/DKO/x_time.npy',np.array(x_time_list))
                np.save('Prediction/DKO/x_all_time.npy',np.array(time_all))
                np.save('Prediction/DKO/x_.npy',np.array(x))
                print('store!!!!')

            return x_pred_list,x_real_list
        ##----------------------------------------------##
        else:
            return self.pred_forward_test_buff(x,u,args)
    
    def pred_forward_test_buff(self,x,u,args): 
        
        pred_horizon = args['old_horizon']
        # loss = nn.MSELoss()
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

        self.net = DKO(args)
        self.net.load_state_dict(torch.load(self.MODEL_SAVE,map_location=self.device))    
        self.net.to(self.device)

        self.optimizer1 = optim.Adam([{'params': self.net.parameters(),'lr':args['lr1'],'weight_decay':args['weight_decay']}])

        print("restore!")


    

class DKO(nn.Module):
    """
    New version of DKO, hope can write better than last version
    """
    def __init__(self, args):
        """A mamba model for LTV system"""
        super().__init__()
        self.L = args['pred_horizon']
        self.O = args['old_horizon']
        self.N = args['latent_dim']
        self.S = args['state_dim']
        self.d_conv = args['d_conv']
        self.delta = args['delta']
        self.P = args['disturbance']
        self.U = args['act_dim']

        self.A = torch.randn(self.N,self.N)*0.01+torch.eye(self.N,self.N)
        self.B = torch.randn(self.N,self.U+self.P)*0.01
        self.C = torch.randn(self.S,self.N)*0.01

        self.A = nn.Parameter(self.A)
        self.B = nn.Parameter(self.B)
        self.C = nn.Parameter(self.C)

        self.x_project_0 = nn.Linear(self.S,args['hidden_dim'])
        self.x_project_1 = nn.Linear(args['hidden_dim'],self.N)

    def forward(self,x, u):
        """
        keep the same form of mamba
        """
        z_0  = self.project_x(x[:,self.O-1,:])
        u_0  = self.project_u(u[:,self.O-1:,:])
        return self.KoopmanOperator(z_0,u_0,x)
    
    def KoopmanOperator(self,z_0,u_all,x):
        ys = []
        loss = nn.MSELoss()  
        loss_ = 0
        for i in range(self.L): # The same as in the controller
            z_0 = einsum('N L, B L -> B N',self.A, z_0)+einsum('N L, B L -> B N',self.B,u_all[:,i,:])
            # ys.append(einsum('N L, B L -> B N',self.C,z_0))
            y = einsum('N L, B L -> B N',self.C,z_0)
            loss_+= loss(y,x[:,self.O+i,:])
            ys.append(y.detach().cpu().numpy())
        return loss_/self.L, np.array(ys)
    
    def project_x(self, x_0):
        """
        Project the state x into high-dimensional space
        """
        return self.x_project_1(F.relu(self.x_project_0(x_0)))
    
    def project_u(self, u):
        """
        Project the input u into high-dimensional space
        """
        # return F.elu(self.u_project(u))
        return u
     

if __name__ == '__main__':
    pass    
