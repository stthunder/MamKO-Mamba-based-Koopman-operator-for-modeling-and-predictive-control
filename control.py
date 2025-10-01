from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
from robustness_eval import *

import argparse
import args_new as new_args

parparser = argparse.ArgumentParser()
parparser.add_argument('method',type=str)
parparser.add_argument('model',type=str)

condition = parparser.parse_args()

def main():
    args = new_args.args
    if condition.model == 'cartpole':
        from envs.cartpole import CartPoleEnv_adv as dreamer
    if condition.model == 'cartpole_V':
        from envs.cartpole_V import CartPoleEnv_adv as dreamer

    args = dict(args,**new_args.ENV_PARAMS[condition.model])
    
    
    env = dreamer()
    env = env.unwrapped  

    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['reference'] = env.xs
    args['method'] = condition.method


    restore_data(env,args)

def recover_model_data_driven(method_choose,env,args):

    print(method_choose.method)
    if method_choose.method == 'mamba':
        from MamKO import Koopman_Desko
        from controller import Upper_MPC_mamba as build_func
    if method_choose.method == 'DKO':
        from DKO import Koopman_Desko
        from controller import Upper_MPC_DKO as build_func
    if method_choose.method == 'MLP':
        from MLP import Koopman_Desko
        from controller import Upper_MPC_MLP as build_func
        args['structure'] = 'MLP'
    
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

    model.shift   = np.loadtxt(args['shift_x'])
    model.shift_u = np.loadtxt(args['shift_u'])
    model.scale   = np.loadtxt(args['scale_x'])
    model.scale_u = np.loadtxt(args['scale_u'])

    args['control'] = True
    model.parameter_restore(args)
    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['s_bound_low'] = env.observation_space.low
    args['s_bound_high'] = env.observation_space.high
    args['a_bound_low'] = env.action_space.low
    args['a_bound_high'] = env.action_space.high
    model.shift_ = [model.shift,model.scale,model.shift_u,model.scale_u]
    args['reference'] = (env.xs-model.shift)/model.scale
    args['reference_'] = env.xs
    controller = build_func(model, args,new_args.ENV_PARAMS[condition.model][method_choose.method]['Q'], new_args.ENV_PARAMS[condition.model][method_choose.method]['R'], 
                            new_args.ENV_PARAMS[condition.model][method_choose.method]['P'])
    return controller


def restore_data(env,args):
    
    args['control']  = True
    args['env_name'] = condition.model

    if condition.method == 'mamba' or condition.method == 'DKO' or condition.method == 'MLP':
        controller = recover_model_data_driven(condition,env,args)
    for i in range(10):
        args['iter'] = i
        dynamic(controller, env, args, condition.method)

    
if __name__ == '__main__':
    main()


