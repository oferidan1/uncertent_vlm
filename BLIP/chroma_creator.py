import argparse
import json
import os
import sys
import os
import pandas as pd
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
import torch

#import numpy as np
#from PIL import Image
from models.blip_retrieval import blip_retrieval
from data import create_dataset, create_sampler, create_loader

import chromadb 
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import ruamel_yaml as yaml
import torch.nn.functional as F

os.environ['WANDB_DISABLED'] = 'true'

def parse_args(args):
    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 48)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--is_resize', default=0, type=int)
    parser.add_argument('--distributed', default=0, type=int)
    parser.add_argument('--is_itc_only', default=0, type=int)
    parser.add_argument('--test_data', default=0, type=int)
    parser.add_argument('--dataset_name', default='coco', type=str, help='Dataset name: coco, vg')
    parser.add_argument('--vlm_model',  default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth', type=str)
    parser.add_argument('--image_collection', default='blip_images', type=str)
    parser.add_argument('--text_collection', default='blip_text', type=str)
    parser.add_argument('--database_path', default='/mnt/d/ofer/vlm/BLIP/chromadb/', type=str)
    parser.add_argument('--config', default='./configs/retrieval_coco.yaml')
    parser.add_argument('--reset', default=1, type=int)

    return parser.parse_args(args)


def get_image_vector(model, image):
    image_feat = model.visual_encoder(image)   
    image_embed = model.vision_proj(image_feat[:,0,:])            
    image_embed = F.normalize(image_embed,dim=-1)      
    return image_embed  

def get_text_vector(model, text, device='cuda'):
    #print("Text")
    text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
    text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
    return text_embed

def db_connect(args):
    db_settings = Settings(chroma_server_host="127.0.0.1",chroma_server_http_port=30303, allow_reset=True , anonymized_telemetry=False)
    
    client = chromadb.PersistentClient(
        path=args.database_path + "/" + args.dataset_name,
        settings=db_settings,
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
        )
    print("Connected to DB: ", args.database_path + "/" + args.dataset_name)
    #client = chromadb.HttpClient(host='127.0.0.1', port=30303, settings=Settings(anonymized_telemetry=False))
    return(client)

def test_chroma_db(args):
    chroma_client = db_connect(args)
    collection = chroma_client.get_or_create_collection(name="my_collection")
    collection.add(
        documents=["doc1", "doc2", "doc3"],
        embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2]],
        metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
        ids=["id1", "id2", "id3"]
    )
    results =collection.query(
        query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2]],
        include=["documents"]
    )
    print(results)


def create_and_save_embeddings(args):
    args = parse_args(args)    
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)        
    
    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, False)  
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[False, False, False], 
                                                          collate_fns=[None,None,None])   
    
    if args.dataset_name == 'coco':
        with open('annotation/coco_karpathy_train.json', 'r') as file:
            ref_data = json.load(file)  
        lookup = pd.DataFrame(ref_data)
    
     #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=args.vlm_model, image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'], distributed=args.distributed, is_itc_only=args.is_itc_only)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()    

    if not args.test_data:
        chroma_client = db_connect(args)
        
        if args.reset:
            reset = "Yes"
        else:
            reset = input("Update of DB not supproted. Please backup db. Do you want to remove old database? [Yes/No] : ")
        if reset == "Yes":
            print("Old database deleted, creating new one...")
            chroma_client.reset()
        else:
            print("New database will not created, exiting...")
            exit()
        image_collection = chroma_client.get_or_create_collection(name=args.image_collection)
        text_collection = chroma_client.get_or_create_collection(name=args.text_collection)
    
    
    if args.test_data:
        for (pix_val, tokens, captions, str_caption, keys, image_idx, urls) in tqdm(train_loader):
            print(pix_val)
        exit()
    else:
        if args.dataset_name == 'coco':
            z = torch.zeros(1)
            for i,(image, captions, idx) in enumerate(train_loader):     
                keys, ids, counts = idx.unique(return_inverse=True, return_counts=True)
                counts_i = counts.cumsum(0)
                image = image[counts_i-1]
                counts_j = torch.cat((z,counts_i),dim=0).int()                
                #create sub list per id from unique ids of captions according to ids
                captions_per_image = [captions[counts_j[j-1]:counts_j[j]] for j in range(1, len(counts_j))]
                keys = keys.cpu().numpy()       
                image_embs = get_image_vector(model, image.to(device)).to('cpu').detach().numpy()
                img_metadatas = []
                for j, (key, img_captions) in enumerate(zip(keys, captions_per_image)):
                    text_metadatas = []
                    text_ids = []
                    text_embs = get_text_vector(model, img_captions).to('cpu').detach().numpy() 
                    # ref_meta = lookup[(lookup['caption'] == cap)]
                    ref_meta = lookup[(lookup['caption'] == img_captions[0])]
                    for i, cap in enumerate(img_captions):
                        text_ids.append(str(key) + "_" + str(i))
                    if not ref_meta.empty:
                        image_id = ("##").join(list(ref_meta.image_id.iloc))
                        image_path = ("##").join(list(ref_meta.image.iloc))
                        meta = {'image_id': image_id, 'url': image_path}
                        for i, cap in enumerate(img_captions):
                            text_metadatas.append(meta)
                        if meta not in img_metadatas:
                            img_metadatas.append(meta)

                        if len(img_captions) == len(text_embs)  == len(text_metadatas) == len(text_ids):     
                            text_collection.add(
                            documents=img_captions,
                            embeddings=text_embs,
                            metadatas=text_metadatas, #[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                            ids=text_ids
                            )
                    else:
                       text_collection.add(
                            documents=img_captions,
                            embeddings=text_embs,
                            # metadatas=[], #[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                            ids=text_ids
                            )
                str_caption = [",".join(sublist) for sublist in captions_per_image]
                keys_list = [str(num) for num in keys]
                if len(str_caption) == len(image_embs) == len(img_metadatas) == len(keys):
                    image_collection.add(
                        documents=str_caption,
                        embeddings=image_embs,
                        metadatas=img_metadatas, #[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                        ids=keys_list
                        )
                else:
                    image_collection.add(
                        documents=str_caption,
                        embeddings=image_embs,
                        #metadatas=[], #[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
                        ids=keys_list
                        )
                    
               
def main(args):
    # vg = load_visual_genome_data("region_descriptions_v1.2.0")
    # for i, (image_region) in enumerate(vg['train']):  
    #     print(image_region)
    #     input() 
    create_and_save_embeddings(args)

if __name__ == '__main__':
  main(sys.argv[1:])