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
        #model=model
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
            #print("yes ref size", batch)
            with torch.no_grad():
                code = model.encode(ref)
                #print(code)
#             codenp=code.detach().cpu().numpy()
#             if it==0:
#                 #it=0
#                 collect_code=codenp
#             else:
#                 collect_code=numpy.append(collect_code,codenp,axis=0)
#             it+=1
        return code,scale,shift

    def reconstruct_pc(args,code,scale,shift):
        it=0
        ckpt = torch.load(args.ckpt)
        seed_all(ckpt['args'].seed)#test_loader=test_loader
        model = AutoEncoder(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        ref=torch.ones([128, 2048, 3])# change this as per
        #batch_size=ref.size(0)
        #test_loader = DataLoader(code, batch_size=args.batch_size, num_workers=args.num_workers) 
        model.eval()
        #for i, batch in enumerate(tqdm(test_loader)):
        #codeb= batch.to(args.device)
        with torch.no_grad():
            recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()
        recons = recons * scale + shift
        reconsnp= recons.detach().cpu()
#         if it==0:
#             all_recons=reconsnp
#         else:
#             all_recons=numpy.append(allrecons,reconsnp,axis=0)
        #all_recons = torch.cat(all_recons, dim=0) 
        return reconsnp
    
#     def ref_pc(args):
#         #test_loader=self.test_loader
#         #collect_code=[]
#         all_ref=[]
#         model=self.model
#         for i, batch in enumerate(tqdm(test_loader)):
#             ref = batch['pointcloud'].to(args.device)
#             shift = batch['shift'].to(args.device)
#             scale = batch['scale'].to(args.device)
        
#             ref = ref * scale + shift
        
#             all_ref.append(ref.detach().cpu())
#         all_ref = torch.cat(all_ref, dim=0)

#         logger.info('Saving ref pcs')    

#         return all_ref



