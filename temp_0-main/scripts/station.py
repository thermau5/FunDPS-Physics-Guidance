import warnings
warnings.filterwarnings("ignore")
import builtins
original_print = builtins.print
def print(*args, **kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    original_print(*args, **kwargs)
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import wandb
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neuralop

import My_TOOL as myt
import torch.fft as fft
from neuralop import get_model
from neuralop.training import setup
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.utils import get_wandb_api_key, count_model_params

import time as tm
from data0.positional_encoding import get_grid_positional_encoding





counter_file = "counter.txt"
file_id=myt.id_filename(counter_file)


##
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ns1w_plot0.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()

###
device, is_logger = setup(config)

arch = config["arch"].lower()
config_arch = config.get(arch)

# PDE setting
import math as mt
'''
###pde information'''
pde_info={}
pde_kf={
    'domain':[[0,2*mt.pi],[0,2*mt.pi]],
    'pde_dim':2,
    'pde_dim_pino':3,
    'function_dim':1,
    'L':[2*mt.pi,2*mt.pi]
}
pde_info['kf']=pde_kf

pde_case=pde_info[config.wandb.pde]
pino_t_tag=0
if 'pino' in config.wandb and config.wandb.pino:
    config_arch.data_channels=pde_case['pde_dim_pino']+pde_case['function_dim']
    config_arch.domain=[[0,config.data.t_predict]]+pde_case['domain']
    if pde_case['pde_dim_pino']!=pde_case['pde_dim']:
        pino_t_tag=1
else:
    config_arch.data_channels=pde_case['pde_dim']+pde_case['function_dim']
    config_arch.domain = pde_case['domain']

'''use wandb, record config'''
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    print("success!!!!")
    print(type(config))
    # if config.wandb.name:
    #     wandb_name = config.wandb.name+f"({file_id})"
    if 1:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                str(file_id),
                str('plot'),

            ]  # +f"({file_id})"
        )

    wandb_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    ###myt
    # myt.ppp(wandb_args)
    # myt.check_dict(config)
    wandb.init(name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
               config={
                   't_prd':config.data.t_predict
               })


    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.finish()

rel_path='/home/thomaslin/FunDPS-Physics-dev-v2/temp_0-main/dataset_and_model/'
dataset={'link': rel_path + "re1000_grid=256_1_N=40_dt=4.0_Ttj=200-320(stat).pt", 'N': 40, 'dtsave': 1 / 4,
      'T': 320, 'res': 256}
# Load dataset
vorticity=torch.load(dataset['link'])


myt.sss(vorticity) #tensor: N,T,X,Y, N is the number of traj; Each snapshot can be used as a valid input. Snapshot={v[n,t] | n, t}

traj_id=2
time_id=5
t_prediction=200+(time_id-1)*dataset['dtsave']
t_pred=config.data.t_predict
time_out_id=time_id+int(t_pred/dataset['dtsave']) # time_id+2

# input and output
dsp=config.data.dsp
x=vorticity[traj_id:traj_id+1,time_id:time_id+1,::dsp,::dsp].unsqueeze(dim=1)
myt.sss(x) # 1,1,1,X,Y: batch, in_channel(function), t,x,y

y=vorticity[traj_id:traj_id+1,time_out_id:time_out_id+1,::dsp,::dsp].unsqueeze(dim=1)




x=x.repeat(1,1,config.data.repeat_ini,1,1).float()#n,1,t,x,y

gridd=get_grid_positional_encoding(x[0], grid_boundaries=config_arch.domain,
                                   dim_pde=config_arch.data_channels-pde_case['function_dim'],channel_dim=0)#1*1*gx*gy*..

"gridd[i]: 1*1*t_repeat*X"

def out_for_ini(x,gridd_tem): #x_shape: (n,1,1,X,y)
    # ss=x.shape
    # gridd_tem=[g.repeat(ss[0],1,1,1)for g in gridd]#n*1*t*x
    x=x.repeat(1,1,config.plot.repeat_ini,1,1)
    return torch.cat([x]+gridd_tem,dim=1)#n,3,t,x,y




config['model']=config_arch
model = get_model(config)
model = model.to(device)

rel_path='/home/thomaslin/FunDPS-Physics-dev-v2/temp_0-main/dataset_and_model/'
model_link=rel_path+'model_[8, 48, 48]_28_32_8064(ep31)_cvt_2.pt'
cpt=torch.load(model_link,map_location=device)
model.load_state_dict(cpt["model"])
del cpt



#Evaluation
model.eval()


xx=x.clone()

ss = x.shape
gridd_tem = [g.repeat(ss[0], 1, 1, 1, 1) for g in gridd]
x=torch.cat([x]+gridd_tem,dim=1)

with torch.no_grad():
    x=x.to(device)
    output=model(x) # 1,1,1,X,Y

final_output=output[...,-1:,:,:]#n,1,1,x,y

myt.sss(final_output) # 1,1,1,X,Y



# Visualization

def tensor_for_draw(x):
    if isinstance(x,torch.Tensor):
        return x.cpu().numpy()
    else:
        return x
def plotheat(x,y,z,figsize=None,xname='x',yname='y',barname='Function Value',title='f(x,y)',label=None,linewidth=1.5,overlap=0,xnum=9,ynum=0,
             vmin=-30,vmax=30,sticksize=30,ftsz=30):
    data_draw = [tensor_for_draw(i) for i in [x,y,z]]


    plt.figure(figsize=figsize)
    # plt.rc('text', usetex=True)

    heatmap = plt.pcolormesh(data_draw[0], data_draw[1], data_draw[2], cmap='bwr', shading='auto',vmin=vmin,vmax=vmax) #cmap=viridis: green=0; bwr: white=0



    # cbar=plt.colorbar(heatmap, label=barname,pad=0.01)
    cbar=plt.colorbar(heatmap,pad=0.02)
    cbar.ax.tick_params(labelsize=ftsz)

    plt.xlabel(xname, fontsize=ftsz)
    plt.ylabel(yname, fontsize=ftsz)
    plt.title(title, fontsize=ftsz)
    plt.xticks(fontsize=sticksize)
    plt.yticks(fontsize=sticksize)



res=int(dataset['res']/dsp)
title_lst=['input','output','truth']
tensor_lst=[xx[:,:,0],final_output,y]
for i in range(3):
    aa=tensor_lst[i].squeeze(0).squeeze(0).squeeze(0)# x,y
    myt.sss(aa)

    plot_x=np.array(list(range(res)))*np.pi/res*2
    plot_y=np.array(list(range(res)))*np.pi/res*2
    plotheat(x=plot_x,y=plot_y,z=aa,title=f'traj={traj_id}, t={t_prediction}, {title_lst[i]}',sticksize=14,ftsz=24)
    name = f'temp_save/visual_{title_lst[i]}'
    plt.tight_layout()
    plt.savefig(name + '.png')
    plt.clf()