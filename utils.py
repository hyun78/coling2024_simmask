"""
edits
- formatting : inserted newlines, spaces etc. (Jun 5)
"""

import random
        
import numpy as np
# 0 1 2 3 42
import sklearn.metrics
import torch
from tqdm import tqdm
import torch.nn.functional as F
from argparse import ArgumentParser
import logging


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Argument parser')

    parser.add_argument('--train_dataset', help='Dataset for training',default='jigsaw',
                        choices=['jigsaw', 'simsimi', 'convabuse', 'sbic', 'bbf'], type=str)
    parser.add_argument('--test_dataset', help='Dataset for evaluation', default='convabuse',
                        choices=['jigsaw', 'simsimi', 'convabuse', 'sbic', 'bbf'], type=str)
    parser.add_argument('--model', help='Model',default='simcse_supcon',
                        choices=['simmask', 'simcse', 'cls', 'bert_finetune', 'tfidf_svm', 'simunk', 'simbad', 'simmask_moco', 'simcse_moco', 'simbad_moco', 'simmask_supcon', 'simcse_supcon', 'simbad_supcon'], type=str) ########
    parser.add_argument('--mode', help='Masking Strategy',
                        default='random',
                        choices=['random','tfidf_offensive','tfidf_both'], type=str)
    parser.add_argument('--bert_name', help='BERT encoder name',
                        default='sbert', type=str) #####change to sbert_2 and see if performance improves
    parser.add_argument('--eval_path', help='Validation performance save path', 
                        default='val_path', type=str) ##############
    parser.add_argument('--model_path', help='Model save path', 
                        default='model_path', type=str) ############
    parser.add_argument("--batch_size", type=int,
                        default=16, help='Batch size')
    parser.add_argument("--num_epochs", type=int,
                        default=5, help='Number of epochs')
    parser.add_argument("--lambda_", type=float,
                        default=0.15, help='Lambda hyperparameter in loss')
    parser.add_argument("--temperature", type=float,
                        default=5e-2, help='temperature hyperparameter in loss')
    parser.add_argument("--lr", type=float,
                        default=3e-5, help='learning rate hyperparameter ')
    parser.add_argument("--masking_ratio", type=float,
                        default=0.3, help='masking ratio hyperparameter')#######
    parser.add_argument("--train_val_size", type=float,
                        default=0.8, help='Train / validation split (train ratio)')
    parser.add_argument("--random_seed", type=int,
                        default=0, help='Random seed')
    parser.add_argument("--K", type=int,
                        default=65536, help='queue size for MoCo')
    parser.add_argument("--others", type=str,
                        default=None, help='other things to add to filename')
    
    
    if default:
        args = parser.parse_args('')  # empty string
        
    else:
        args = parser.parse_args()
    
    if args.bert_name == 'sbert':
        args.bert = "sentence-transformers/bert-base-nli-mean-tokens"
        
    elif args.bert_name == 'sbert_2':
        args.bert = "sentence-transformers/all-mpnet-base-v2"
        
    return args



def get_res(model, dataloader, record_misclassification=False):
    
    all_outputs = get_outputs(model, dataloader)

    return evaluate(dataloader, all_outputs, record_misclassification=record_misclassification)
    
    
def get_outputs(model, dataloader, ce=True):
    
    tbar= tqdm(dataloader)
    all_outputs = []

    all_text = []
    
    for inputs, targets, _ in tbar:

        input_ids = inputs['input_ids'].long().cuda()
        attention_mask = inputs['attention_mask'].long().cuda()
        targets = targets.long().cuda()
        output = model(input_ids=input_ids, attention_mask=attention_mask, ce=ce)
        all_outputs.append(output.cpu())

    all_outputs = torch.cat(all_outputs)
    
    return all_outputs


def evaluate(dataloader, all_outputs, record_misclassification=False):
    
    pred = all_outputs.argsort(dim=-1, descending=True)[:, :1]
    pred = pred.squeeze().numpy()

    target = dataloader.dataset.targets[:pred.shape[0]]

    correct = torch.Tensor(pred == target)

#     target = 1-target
    acc = correct.sum().item() / pred.shape[0]
    f1 = sklearn.metrics.f1_score(target, pred, average='macro')
    prec = sklearn.metrics.precision_score(target, pred, average='macro')
    recall = sklearn.metrics.recall_score(target, pred, average='macro')
#     acc = sklearn.metrics.accruacy_score(target,pred,average='weighted')

    if record_misclassification:
        return (prec, recall, f1, acc), correct
    else:
        return (prec, recall, f1, acc)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class InfoNCE():
#     def __init__(self,temperature,batch_size):
#         self.temperature = temperature
#         self.batch_size = batch_size
#     def __call__(self,out,do_normalize=True):
#         if do_normalize:
# #             out = F.normalize(out,dim=1)
#             out = F.normalize(out,dim=1,p=2)
#         batch_size = int(out.shape[0]/2)
#         if batch_size!=self.batch_size:
#             bs = batch_size
#         else:
#             bs = self.batch_size
#         out_1,out_2 =  out.split(bs,dim=0) # (B,D) , (B,D)
#         align_loss = -torch.sum(out_1 * out_2, dim=-1).mean() # Cosine Sim
#         return align_loss


class InfoNCE():
    
    def __init__(self,temperature,batch_size):
        self.temperature = temperature
        self.batch_size = batch_size
        
    def __call__(self,out,do_normalize=True):
        
        # out: concatenated data (original + positive)
        
        if do_normalize:
#             out = F.normalize(out,dim=1)
            out = F.normalize(out,dim=1,p=2)
        batch_size = int(out.shape[0]/2)
        
        if batch_size!=self.batch_size:
            bs = batch_size
        else:
            bs = self.batch_size
            
        out_1,out_2 =  out.split(bs,dim=0) # (B,D) , (B,D) 
#         out_2 = out_2.detach()
    # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
#         # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

#         # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
#         # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        return loss
    
    
def train_test_split(data,train,test,val):
    
    N = data.shape[0]
    train,test,val = int(N*train), int(N*test), int(N*val)
    train = N-(train+test+val)+train
    assert train+test+val == N
    
    shuffled_data = data.sample(frac=1.0).reset_index()
#     cls = ['implicit_hate', 'not_hate']
#     shuffled_data['targets'] = shuffled_data['class'].apply(lambda x: cls.index(x))

    train_data = shuffled_data.loc[:train-1]
    test_data = shuffled_data.loc[train:train+test]
    val_data = shuffled_data.loc[train+test:]
    
    
    train_corpus, train_targets = train_data['post'].values, np.stack(train_data['targets'].values)
    test_corpus, test_targets = test_data['post'].values, np.stack(test_data['targets'].values)
    val_corpus, val_targets = val_data['post'].values, np.stack(val_data['targets'].values)
    
    return train_corpus, train_targets, test_corpus, test_targets ,val_corpus, val_targets

def logging_config(path="log/", filename="default.log"):
    logging.basicConfig(filename=path+filename, level=logging.INFO)
    
def log(msg):
    logging.info(str(msg))

def load_bad_words():
    path = "./bad_word_list.txt"
    bad_words = []
    with open(path, 'r') as file:
        for line in file:
            bad_words.append(line.strip())
        
        return bad_words
    
