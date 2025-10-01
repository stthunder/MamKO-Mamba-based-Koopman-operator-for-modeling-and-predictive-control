from replay_fouling import ReplayMemory
# from variant import *

# from three_tanks import three_tank_system as dreamer
# from MBR import MBR as dreamer

from torch.utils.data import Dataset, DataLoader, random_split
import gym

import numpy as np
import torch
import os

import argparse


parparser = argparse.ArgumentParser()
parparser.add_argument('method',type=str)
parparser.add_argument('model',type=str)

condition = parparser.parse_args()


def main():     
    import args_new as new_args
    args = new_args.args

    if condition.model == 'cartpole':
        from envs.cartpole import CartPoleEnv_adv as dreamer
    if condition.model == 'cartpole_V':
        from envs.cartpole_V import CartPoleEnv_adv as dreamer
        
    if condition.model == 'half_cheetah':
        # from envs.half_cheetah_cost import HalfCheetahEnv_cost as dreamer
        dreamer = gym.make('HalfCheetah-v2')
    args['env'] = condition.model

    

    args = dict(args,**new_args.ENV_PARAMS[condition.model])


    if condition.method == 'mamba':
        from MamKO import Koopman_Desko
        args['method'] = 'mamba'
    if condition.method == 'DKO':
        from DKO import Koopman_Desko
        args['method'] = 'DKO'
    if condition.method == 'MLP' or condition.method == 'LSTM' or condition.method == 'TRANS':
        from MLP import Koopman_Desko
        args['method'] = 'MLP'
        args['structure'] = condition.method
        # args['optimize_step'] = 20

    args['continue_training'] = True
    
    for i in range(10):
        env = dreamer()
        env = env.unwrapped  
        args['state_dim'] = env.observation_space.shape[0]
        args['act_dim'] = env.action_space.shape[0]
        args['control'] =  False

        fold_path = 'save_model/'+condition.method+'/'+str(condition.model)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        args['save_model_path'] = fold_path+'/model.pt'
        args['save_opti_path']  = fold_path+'/opti.pt'
        args['shift_x']         = fold_path+'/shift_x.txt'
        args['scale_x']         = fold_path+'/scale_x.txt'
        args['shift_u']         = fold_path+'/shift_u.txt'
        args['scale_u']         = fold_path+'/scale_u.txt'
        
        model = Koopman_Desko(args)
        args['times_training'] = i
        train(args,model,env,i)

        if not args['continue_training']:
            break

def train(args,model,env,i):
    # print(condition.model)
    if not args['import_saved_data']:
        # model.parameter_restore(args)
        replay_memory = ReplayMemory(args,env, predict_evolution=True)
        #############################00000000000#########################
        x_train = replay_memory.dataset_train
        #############################00000000000#########################
        x_val = replay_memory.val_subset
        x_test = replay_memory.dataset_test
        test_draw = replay_memory.dataset_test_draw
    
    #
    else:
        x_train   = torch.load(args['SAVE_TRAIN'])
        x_val     = torch.load(args['SAVE_VAL'])
        x_test    = torch.load(args['SAVE_TEST'])
        test_draw = torch.load(args['SAVE_DRAW'])

    ##-------------------是否使用之前参数重新训练------------------##
    args['restore'] = False 
    ##-----------------------------------------------------------##
    if args['restore'] == True:
        model.parameter_restore(args)
        # test_draw = torch.load(args['SAVE_DRAW'])

    test_data = DataLoader(dataset = test_draw, batch_size = 1, shuffle = True, drop_last = False)

    loss = []
    loss_t = []
    for e in range(args['num_epochs']):
        model.learn(e,x_train,x_val,x_test,args)
        if(e%10==0):
            model.parameter_store(args)
        
        if(e%50==0):
            for x,u in test_data:
                _,_ = model.pred_forward_test(x.float(),u.float(),True,args,e)
        loss.append(model.loss_store)
        loss_t.append(model.loss_store_t)   
        

    fold_path = 'loss/'+condition.method+'/'+str(condition.model)+'/'+str(i)
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
        
    fold_path_ = fold_path+'/loss_.txt'
    np.savetxt(fold_path_,np.array(loss))
    fold_path_ = fold_path+'/loss_t.txt'
    np.savetxt(fold_path_,np.array(loss_t))  

            
                 


if __name__ == '__main__':
    main()