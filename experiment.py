#import os
import time
import argparse
import torch
from tqdm.auto import tqdm
import numpy 

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import *
from latent_space import * #this contains latentspace class

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/chair.pt')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./shape_net_core_uniform_samples_2048')
parser.add_argument('--batch_size', type=int, default=339) #please update this to the number of pcs
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

#code is the latent space
#scale and shift is neede as inputs to the decoder
#ref is the scaled and shifted original pcs needed for evaluation

code,scale,shift,ref=lspace_experiments.create_latent_space(args)

#recons is the reconstructed pcs(scaled and shifted)
recons=lspace_experiments.reconstruct_pc(args,code,scale,shift)





# with torch.no_grad():
#     results = compute_all_metrics(recons, ref, args.batch_size, accelerated_cd=True)
#     results = {k:v.item() for k, v in results.items()}
#     jsd = jsd_between_point_cloud_sets(recons.cpu().numpy(), ref.cpu().numpy())
#     results['jsd'] = jsd
# print(results)

# metrics = EMD_CD(recons, ref, batch_size=args.batch_size, accelerated_cd=True)
# print(metrics)
# ref_file='ref.npy'
# rec_file='rec.npy'

# numpy.save(ref_file, ref)

# numpy.save(rec_file, recons)