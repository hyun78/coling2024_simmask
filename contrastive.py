from augmentation import *
from utils import load_bad_words
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

class OffLangContrastivedataset(Dataset):

    def __init__(self, bert,original_texts, implied_texts, max_len = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.org = []
        self.pos = []
        self.max_len = max_len
#         self.targets = targets

        assert len(original_texts) == len(implied_texts)
    
        for idx in tqdm(range(len(implied_texts))):
            org = original_texts[idx]
            pos = implied_texts[idx]
            org_input = self.tokenizer(org, padding='max_length', truncation=True, 
                                       max_length=self.max_len, return_tensors='pt')
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            
            pos_input = self.tokenizer(pos, padding='max_length', truncation=True, 
                                       max_length=self.max_len, return_tensors='pt')
            pos_input['input_ids'] = torch.squeeze(pos_input['input_ids'])
            pos_input['attention_mask'] = torch.squeeze(pos_input['attention_mask'])
            self.org.append(org_input)
            self.pos.append(pos_input)

    def __len__(self):
        return len(self.org)

    def __getitem__(self, idx):
        return self.org[idx],self.pos[idx]

def load_contrastive_dataset(args, train_corpus, train_targets=None):
    
    if 'simmask' in args.model:
        
        masked_corpus = masking(args, train_corpus)
#         replaced_corpus = token_replacement(args, train_corpus, '[MASK]')
                                      
        cont_trainds = OffLangContrastivedataset(bert=args.bert,original_texts=train_corpus,implied_texts = masked_corpus)
        cont_trainloader = DataLoader(cont_trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)

    elif 'simcse' in args.model:
        
        cont_trainds = OffLangContrastivedataset(bert=args.bert,original_texts=train_corpus,implied_texts = train_corpus)
        cont_trainloader = DataLoader(cont_trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)
        
        
    elif 'simunk' in args.model:
        
        replaced_corpus = token_replacement(args, train_corpus, '[UNK]')
                                      
        cont_trainds = OffLangContrastivedataset(bert=args.bert,original_texts=train_corpus,implied_texts = replaced_corpus)
        cont_trainloader = DataLoader(cont_trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)

    elif 'simbad' in args.model:

        bad_words = load_bad_words()

        augmentation = token_conditional_insertion_one(args, train_corpus, bad_words, targets=train_targets)
        
        cont_trainds = OffLangContrastivedataset(bert=args.bert,original_texts=train_corpus,implied_texts = augmentation)
        cont_trainloader = DataLoader(cont_trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)
        
        
    else:
        cont_trainloader = None
        
    return cont_trainloader