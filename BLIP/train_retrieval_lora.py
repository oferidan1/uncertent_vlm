'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel

def sattolos_shuffle(indices):
    """
    Shuffles a list of indices in-place using Sattolo's algorithm,
    ensuring no element remains in its original position.
    """
    n = len(indices)
    for i in range(n - 1, 0, -1):
        # Choose a random index j from 0 to i-1 (inclusive)
        j = random.randrange(i)
        # Swap indices[i] and indices[j]
        indices[i], indices[j] = indices[j], indices[i]
    return indices

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    #triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    # Custom Distance Function (Lambda)
    triplet_loss = (nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x,y)))
    loss_itm = torch.tensor(0.0, device=device)
    loss_ita = torch.tensor(0.0, device=device)

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        loss = 0.0
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        if args.loss_type == 1:  # ITC only            
            loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)         
            loss = loss_ita       
        if args.loss_type == 2:  # ITC + ITM
            # blip original loss
            loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)         
            loss = loss_ita + loss_itm
        elif args.loss_type == 3:  # Triplet Loss
            # blip triplet loss
            text_embeds  = model.encode_text(caption)            
            image_embeds = model.encode_image(image)     
            batch_size = text_embeds.size(0)
            indices = torch.arange(batch_size, device=device)
            # do triplet loss with different negative samples
            for j in range(batch_size//4):
                indices_shuffled = sattolos_shuffle(indices.clone())
                neg_image_embeds = image_embeds[indices_shuffled]                   
                loss_ita = triplet_loss(text_embeds, image_embeds, neg_image_embeds)
                loss += loss_ita
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()           
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader, device, config, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    
    debug = False
    
    # compute text features
    print('Computing text features...')
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
        
        if debug:
            break
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id
    
    # compute image features
    print('Computing image features...')
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed,dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
        
        if debug:
            break
     
    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    
    # if itc only, return the similarity matrix
    if args.is_itc_only:
        return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()
    
    score_matrix_i2t = torch.full((len(data_loader.dataset.image),len(texts)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx.cpu()].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
        
    config['batch_size_train'] = args.batch_size_train
    config['batch_size_test'] = args.batch_size_test    
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=args.pretrained, image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'], distributed=args.distributed, is_itc_only=args.is_itc_only)

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module   
        
    if args.lora:        
        # loop pver model parameters and set requires_grad=False
        for param in model.parameters():
            param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_rslora=True,
            #target_modules=["qkv"],
            target_modules=["query", "key", "value", "qkv", "proj", "text_proj", "vision_proj", "text_proj_m", "vision_proj_m"],
            #task_type=TaskType.SEQ_CLS, # ???
        )
        
        model = get_peft_model(model_without_ddp, lora_config)
        model.print_trainable_parameters() # see % trainable parameters

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    model.to(device)
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, train_loader, optimizer, epoch, device, config)  
            
        if args.eval_on == 'val' or args.eval_on == 'both':
            score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config, args)
        if args.eval_on == 'test' or args.eval_on == 'both':
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config, args)
    
        if utils.is_main_process():  
            
            if args.eval_on == 'val' or args.eval_on == 'both':     
                val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
                print(val_result)
                                
            if args.is_train and val_result['r_mean']>best:
                if args.lora:
                    model.save_pretrained(args.output_dir)
                else:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                    best = val_result['r_mean']        
                    best_epoch = epoch  
                
            if args.eval_on == 'test' or args.eval_on == 'both':
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                print(test_result)
            
            if args.evaluate:                
                if args.eval_on == 'val':
                    log_stats = {**{f'val_{k}': v for k, v in val_result.items()}, }   
                elif args.eval_on == 'test':
                    log_stats = {**{f'test_{k}': v for k, v in test_result.items()}, }
                else:
                    log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
        if args.evaluate: 
            break

        if args.distributed:
            # wait for all processes to finish before moving to the next epoch
            dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    # parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    # parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--config', default='./configs/retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_coco')    
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=0, type=int)
    parser.add_argument('--lora', default=0, type=int)
    parser.add_argument('--batch_size_train', default=24, type=int)
    parser.add_argument('--batch_size_test', default=256, type=int)
    parser.add_argument('--is_train', default=1, type=int)
    parser.add_argument('--is_itc_only', default=1, type=int)    
    parser.add_argument('--loss_type', default=3, type=int, choices=['1', '2', '3'], help='1: ITC only, 2: ITC + ITM, 3: Triplet Loss')    
    parser.add_argument('--pretrained', default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth')    
    parser.add_argument('--eval_on', default='both', type=str, choices=['val', 'test', 'both'], help='which set to evaluate on')
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)