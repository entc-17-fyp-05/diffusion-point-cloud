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
from evaluation import EMD_CD


class lspace_experiments():
    
    def create_latent_space(args):  
        it=0
        ckpt = torch.load(args.ckpt)
        seed_all(ckpt['args'].seed)#test_loader=test_loader
        model = AutoEncoder(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        collect_code=[]
        latent_dset = ShapeNetCore(
            path=args.dataset_path,
            cates=args.categories,
            split='val',
            scale_mode=ckpt['args'].scale_mode
        )
        test_loader = DataLoader(latent_dset, batch_size=args.batch_size, num_workers=args.num_workers) 
        for i, batch in enumerate(tqdm(test_loader)):
            ref = batch['pointcloud'].to(args.device)
            shift = batch['shift'].to(args.device)
            scale = batch['scale'].to(args.device)
            model.eval()
            with torch.no_grad():
                code = model.encode(ref)
            ref = ref * scale + shift
            all_ref=ref.detach().cpu().numpy()
        return code,scale,shift,all_ref

    def reconstruct_pc(args,code,scale,shift):
        it=0
        ckpt = torch.load(args.ckpt)
        seed_all(ckpt['args'].seed)#test_loader=test_loader
        model = AutoEncoder(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        ref=torch.ones([128, 2048, 3])# change this as per
        model.eval()
        with torch.no_grad():
            recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()
        recons = recons * scale + shift
        reconsnp= recons.detach().cpu().numpy()
        return reconsnp
    


