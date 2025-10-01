import numpy as np
args = {
'batch_size': 256,
'import_saved_data': False,
'val_frac': 0.2,
'lr1': 0.001,
'gamma': 0.9,
'num_epochs' : 401,
'weight_decay': 0.001,

'SAVE_TEST' : "/test.pt",
'SAVE_TRAIN' : "/train.pt",
'SAVE_VAL' : "/val.pt",
'SAVE_DRAW' : "/draw.pt",

'apply_state_constraints': False,
'apply_action_constraints': True,
}

ENV_PARAMS = {
'cartpole':
{
    'pred_horizon': 30,
    'control_horizon':30,
    'old_horizon':30,
    'latent_dim': 8, 
    'd_conv' : 10,
    'delta': 10,
    'total_data_size': 40000-100,
    'total_data_size_test': 4000,
    'max_ep_steps': 20000+40,
    'max_ep_steps_test':1000,
    'optimize_step':50,
    'loop_test':1000,
    'hidden_dim':64,
    'disturbance' : 0,
    'mamba':{
        'Q':np.diag([1,0.01,100,0.01]),
        'R':np.diag([0.5]),
        'P':np.diag([5000,0,0,0]),
    },
    'DKO':{
        'Q':np.diag([1,0.01,100,0.01]),
        'R':np.diag([0.5]),
        'P':np.diag([5000,0,0,0]),
    },
    'MLP':{
        'Q':np.diag([1,0.01,20,0.01]),
        'R':np.diag([0.5]),
        'P':np.diag([5000,0,0,0]),
    },
},
'cartpole_V':
{
    'pred_horizon': 30,
    'control_horizon':30,
    'old_horizon':30,
    'latent_dim': 8, 
    'd_conv' : 15,
    'delta': 5,
    'hidden_dim':64,
    'import_saved_data': False,
    'total_data_size': 40000-100,
    'total_data_size_test': 4000,
    'max_ep_steps': 20000+40,
    'max_ep_steps_test':1000,
    'optimize_step':50,
    'loop_test':1000,
    'disturbance' : 0,
    'mamba':{
        'Q':np.diag([1,0.0001,100,0.0001]),
        'R':np.diag([0.1]),
        'P':np.diag([5000,0,0,0]),
    },
    'DKO':{
        'Q':np.diag([1,0.0001,100,0.0001]),
        'R':np.diag([0.1]),
        'P':np.diag([2000,0,0,0]), # TODO: TOO LARGE MAY FAIL
    },
    'MLP':{
        'Q':np.diag([1,0.0001,100,0.0001]),
        'R':np.diag([0.1]),
        'P':np.diag([2000,0,0,0]), # TODO: TOO LARGE MAY FAIL
    },
}
}