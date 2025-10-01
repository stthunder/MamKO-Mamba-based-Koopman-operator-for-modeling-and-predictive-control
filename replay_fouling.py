import math
import numpy as np
import random
import progressbar
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import json
import pandas as pd




class MyDataSet(Dataset):

    def __init__(self, test, path = None, Load = False, x = None, y = None):
        if Load:
            """
            load data from the specific path
            """
            self.data = np.loadtxt(path)  
            """
            TODO: save_data
            """
            self.x = self.data[:, 1:]  
            self.y = self.data[:, 0] 
        else:
            #simple input
            """
            from generating
            """  
            self.x = x  
            self.u = y 
        self.test = test



    def __len__(self):
        return len(self.x_choice)

    def __getitem__(self, index):
        return self.x_choice[index, :], self.y_choice[index]
    
    def determine_shift_and_scale(self,args):
        # maybe not so precise since the data may be drop by dataloader
        self.shift_x = np.mean(self.x, axis=(0, 1))
        self.scale_x = np.std(self.x, axis=(0, 1))
        self.shift_u = np.mean(self.u, axis=(0, 1))
        self.scale_u = np.std(self.u, axis=(0, 1))

        self.scale_x[np.argwhere(self.scale_x==0)]=1
        self.scale_u[np.argwhere(self.scale_u==0)]=1

        np.savetxt(args['shift_x'],self.shift_x)
        np.savetxt(args['shift_u'],self.shift_u)
        np.savetxt(args['scale_x'],self.scale_x)
        np.savetxt(args['scale_u'],self.scale_u)

        return [self.shift_x,self.scale_x,self.shift_u,self.scale_u]
    
    def shift_scale(self, shift_ = None):

        if self.test:
            self.x_choice = (self.x - shift_[0])/shift_[1]
            self.y_choice = (self.u - shift_[2])/shift_[3]
            # zero

        else:
            self.x_choice = (self.x - self.shift_x)/self.scale_x
            self.y_choice = (self.u - self.shift_u)/self.scale_u

# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, env, predict_evolution = False, LSTM = False):
        """Constructs object to hold and update training/validation data.
        Args:
            args: Various arguments and specifications
            shift: Shift of state values for normalization
            scale: Scaling of state values for normalization
            shift_u: Shift of action values for normalization
            scale_u: Scaling of action values for normalization
            env: Simulation environment
            net: Neural network dynamics model
            sess: TensorFlow session
            predict_evolution: Whether to predict how system will evolve in time
        """
        self.batch_size = args['batch_size']
        self.seq_length = args['pred_horizon']
        self.seq_length_O = args['old_horizon']
        self.env = env
        self.total_steps = 0
        self.LSTM = LSTM

        self.us = np.array([240,90000])



        if args['import_saved_data']:
            self._restore_data('./data/' + args['env_name'])

        else:
            print("generating data...")
            self._generate_data(args)

    
    def _generate_train(self, args, if_train):

        # Initialize array to hold states and actions
        x = []
        u = []

        if if_train:
            length_ = args['total_data_size']
        else:
            length_ = args['total_data_size_test']
        print(length_)
        # Define progress bar
        bar = progressbar.ProgressBar(maxval=length_).start()

        # Loop through episodes
        while True:
            t = 0
            start_number = 0
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
            u_trial = np.zeros((args['max_ep_steps']-1, args['act_dim']+args['disturbance']), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[t] = self.env.reset()


            # for t in range(1, args['max_ep_steps']):
            while(True):
                # print(t)
                action = self.env.get_action()
                input = action

                u_trial[t] = input
                _,_,done,_ = self.env.step(input)
                t+=1
                x_trial[t] = self.env.x
                if done or t==args['max_ep_steps']-1:
                    break
                # print(t)
            # length_list.append(t)
            #-------------------------------For the structure of designed mamba-----------------------------#
            add_ = np.repeat(x_trial[start_number,:].reshape(-1,1),self.seq_length_O,axis =1) # For the first state !
            x_trial_ = np.concatenate((add_.T,x_trial[start_number:t,:]))
           
            add_ = np.repeat(np.zeros_like(u_trial[start_number,:].reshape(-1,1)),self.seq_length_O,axis =1) # TODO: before no control input 
            u_trial_ = np.concatenate((add_.T,u_trial[start_number:t-1,:]))
            j = self.seq_length_O
            #-----------------------------------------------------------------------------------------------#

            while j + self.seq_length < len(x_trial_):
                x.append(x_trial_[j - self.seq_length_O:j + self.seq_length])
                u.append(u_trial_[j - self.seq_length_O:j + self.seq_length-1])
                # j+=np.random.randint(3,high=10,size=1)[0]  #TODO: +=1
                j+=1
                
            # print("???")
            if len(x) >= length_:
                break
            bar.update(len(x))
        bar.finish()

        x = np.array(x)
        u = np.array(u)

        len_x = int(np.floor(len(x)/args['batch_size'])*args['batch_size'])
        x = x[:len_x]
        u = u[:len_x]

        
        if if_train:
            self.dataset_train = MyDataSet(test = False, x = x, y = u)
            self.shift_ = self.dataset_train.determine_shift_and_scale(args)
            self.dataset_train.shift_scale()

            len_train = len(self.dataset_train)
            len_val = int(np.round(len_train*args['val_frac']))
            len_train -= len_val
            self.train_subset, self.val_subset = random_split(self.dataset_train,[len_train, len_val],generator=torch.Generator().manual_seed(1))
        
        else:
            self.dataset_test =  MyDataSet(test = True, x = x, y = u)
            self.dataset_test.shift_scale(self.shift_)

    def _generate_draw(self,args):

        while(True):
            t = 0
            x_test = []
            u_test = []
            x_test.append(self.env.reset())
            for t in range(1, args['max_ep_steps_test']):
                action = self.env.get_action()
                input = action

                _,_,done,_ = self.env.step(input)

                u_test.append(input)
                x_test.append(self.env.x)

                if done:
                    break

            if t>self.seq_length+5:
                break

        x = np.array(x_test)
        u = np.array(u_test)


        #-------------------------------For the structure of designed mamba-----------------------------#
        add_ = np.repeat(x[0,:].reshape(-1,1),self.seq_length_O,axis =1) # For the first state !
        x = np.concatenate((add_.T,x))

        add_ = np.repeat(np.zeros_like(u[0,:].reshape(-1,1)),self.seq_length_O,axis =1) # For the first state !
        u = np.concatenate((add_.T,u))
        #-----------------------------------------------------------------------------------------------#

        x_test = x.reshape(-1, x.shape[0], args['state_dim'])
        u_test = u.reshape(-1, u.shape[0], args['act_dim']+args['disturbance'])

        self.dataset_test_draw =  MyDataSet(test = True, x = x_test, y = u_test)
        self.dataset_test_draw.shift_scale(self.shift_)
        



    def _generate_data(self, args):
        """Load data from environment
        Args:
            args: Various arguments and specifications
        """

        self._generate_train(args, True)
        self._generate_draw(args)
        self._generate_train(args, False)
        

    def _store_test(self):
        pass

    def update_data(self, x_new, u_new, val_frac):
        """Update training/validation data
        TODO:
        Args:
            x_new: New state values
            u_new: New control inputs
            val_frac: Fraction of new data to include in validation set
        """
        pass

    def save_data(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(path + '/x.npy', self.x)
        np.save(path + '/u.npy', self.u)
        np.save(path + '/x_test.npy', self.x_test)
        np.save(path + '/u_test.npy', self.u_test)
        np.save(path + '/x_val.npy', self.x_val)
        np.save(path + '/u_val.npy', self.u_val)

    def _restore_data(self, path):
        self.x = np.load(path + '/x.npy')
        self.u = np.load(path + '/u.npy')
        self.x_val = np.load(path + '/x_val.npy')
        self.u_val = np.load(path + '/u_val.npy')
        self.x_test = np.load(path + '/x_test.npy')
        self.u_test = np.load(path + '/u_test.npy')
