from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn
import json
import random

class OffLangdataset(Dataset):
    
    def __init__(self, bert, targets, text_list, max_len = 512):
        
        """
        bert(str): BERT name as string; default: "sentence-transformers/bert-base-nli-mean-tokens"
        targets(list of int): Offensive lang label (0:normal, 1:offensive)
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.data = []
        self.max_len=max_len
        self.targets = targets
        self.orig_text = text_list

        for text in tqdm(text_list):
            
            org_input = self.tokenizer(text, padding='max_length', truncation=True, 
                                       max_length=self.max_len, return_tensors='pt')
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            self.data.append(org_input)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.orig_text[idx]

'''

class OffLangTestdataset(Dataset):

    def __init__(self, bert, targets, text_list, max_len = 512):
        
        """
        bert(str): BERT name as string; default: "sentence-transformers/bert-base-nli-mean-tokens"
        targets(list of int): Offensive lang label (0:normal, 1:offensive)
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert)
        self.data = []
        self.max_len=max_len
        self.targets = targets
        self.orig_text = text_list

        for text in tqdm(text_list):
            
            org_input = self.tokenizer(text, padding='max_length', truncation=True, 
                                       max_length=self.max_len, return_tensors='pt')
            org_input['input_ids'] = torch.squeeze(org_input['input_ids'])
            org_input['attention_mask'] = torch.squeeze(org_input['attention_mask'])
            self.data.append(org_input)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.orig_text[idx]

'''

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
    
    

def train_test_split(data,train,test,val):
    
    N = data.shape[0]
    test, val = int(N * test), int(N * val) #
    train = N - (test + val) #
    assert train+test+val == N
    
    # for random split: shuffle and slice data
    shuffled_data = data.sample(frac=1.0).reset_index()
    train_data = shuffled_data.loc[:train] 
    test_data = shuffled_data.loc[train:train+test]
    val_data = shuffled_data.loc[train+test:]

    return train_data, test_data, val_data


def get_preprocessed_dataframe(args, mode='train'):

    if mode == 'train':

        df = load_dataframe(args.train_dataset, mode='train')
        extract_post_targets(args.train_dataset, df)
        
    elif mode == 'valid':
        
        df = load_dataframe(args.train_dataset, mode='valid')
        extract_post_targets(args.train_dataset, df)

    elif mode == 'test':

        df = load_dataframe(args.test_dataset, mode='test')
        extract_post_targets(args.test_dataset, df)
        
    # extract the main text(post) and targets (offensive = 1, non-offensive = 0) as columns
    
    
    return df


def get_train_corpus_targets(args):
        
    split_valid_set = ['jigsaw', 'simsimi'] # datasets that have a separate test set but not a separate validation set

    df = get_preprocessed_dataframe(args, mode='train')
    
    # sample from non-offensive items so that offensive:non-offensive = 1:1
    df = _balance_dataset(df)
    
    # train-valid split
    
    if args.train_dataset in split_valid_set:
        #train_corpus, train_targets, _, _, val_corpus, val_targets = train_test_split(df, args.train_val_size, 0, 1-args.train_val_size)
        train_data, _, valid_data = train_test_split(df, args.train_val_size, 0, 1-args.train_val_size)
        train_corpus, train_targets = train_data['post'].values, train_data['targets'].values
        val_corpus, val_targets = valid_data['post'].values, valid_data['targets'].values

    else:
        val_df = get_preprocessed_dataframe(args, mode='valid')
        
        train_corpus, train_targets = df['post'].values, df['targets'].values
        val_corpus, val_targets = val_df['post'].values, val_df['targets'].values
    

    return train_corpus, train_targets, val_corpus, val_targets


def load_test_dataset(args):
    
    df = get_preprocessed_dataframe(args, mode='test')

    test_corpus, test_targets = df['post'].values, df['targets'].values
    
    testds = OffLangdataset(bert=args.bert, targets=test_targets, text_list=test_corpus)
    testloader = DataLoader(testds, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)

    return testloader



def load_dataframe(dataset_name, mode='train'):
    
    if dataset_name == "jigsaw":
        
        if mode == 'train':
            path = '../Simsim/benchmark_dataset/Jigsaw/train.csv'
            df = pd.read_csv(path)
            
        if mode == 'valid':
            raise
            
        elif mode == 'test':
            
            path = '../Simsim/benchmark_dataset/Jigsaw/test_labels.csv'
        
            df = pd.read_csv(path)
            df_text = pd.read_csv('../Simsim/benchmark_dataset/Jigsaw/test.csv')
            df = df[df['toxic']!=-1].set_index('id')
            df['comment_text'] =  df_text.set_index('id')['comment_text']

    elif dataset_name == "jigsaw_bugged" or dataset_name == "jigsaw_buggable":
    
        if mode == "train" or mode == "valid":
            raise

        elif mode == "test":

            path = "bugged_dataset/jigsaw_bugged.csv"

            df = pd.read_csv(path)

    elif dataset_name == 'convabuse':
        
        if mode == 'train':
        
            path = '../Simsim/benchmark_dataset/ConvAbuse/convabuse/2_splits/ConvAbuseEMNLPtrain.csv'

            df = pd.read_csv(path, keep_default_na=False)
            
        elif mode == 'valid':
            
            path = '../Simsim/benchmark_dataset/ConvAbuse/convabuse/2_splits/ConvAbuseEMNLPvalid.csv'

            df = pd.read_csv(path, keep_default_na=False)
            
            
        elif mode == 'test':
            
            path = '../Simsim/benchmark_dataset/ConvAbuse/convabuse/2_splits/ConvAbuseEMNLPtest.csv'
        
            df = pd.read_csv(path, keep_default_na=False)
    
    elif dataset_name == 'simsimi':
        
        if mode == 'train':
            
            path = '../Simsim/benchmark_dataset/Simsimi/teaching_train.pkl'
            
        elif mode == 'valid':
            
            raise #TODO
            
        elif mode == 'test':
            
            path = '../Simsim/benchmark_dataset/Simsimi/teaching_test.pkl'
        
        df = pd.read_pickle(path)
        
        
    elif dataset_name == 'sbic':
        
        if mode == 'train':
            
            path = '../Simsim/benchmark_dataset/SocialBias/SBIC.v2.agg.trn.csv'

        elif mode == 'valid':
            
            path = '../Simsim/benchmark_dataset/SocialBias/SBIC.v2.agg.dev.csv'
        
        elif mode == 'test':
        
            path = '../Simsim/benchmark_dataset/SocialBias/SBIC.v2.agg.tst.csv'

        df = pd.read_csv(path)

    elif dataset_name == 'bbf':

        path = "../Simsim/benchmark_dataset/BFB/single_turn_safety.json"

        dfs = []

        with open(path) as file:

            data = json.load(file)

            

            for round in ['1', '2', '3']:

                good = data['standard'][mode][round]['good']
                good_df = pd.DataFrame.from_dict(good)
                good_df['targets'] = 0

                dfs.append(good_df)

                for t in ['adversarial', 'standard']:

                    bad = data[t][mode][round]['bad']
                    bad_df = pd.DataFrame.from_dict(bad)
                    bad_df['targets'] = 1

                    dfs.append(bad_df)
            
        df = pd.concat(dfs)
                        
    return df
    


def extract_post_targets(dataset_name, df):
    
    if dataset_name == "jigsaw":
        
        offensive_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # can be changed
        df['targets'] = df.apply(lambda x : max([x[c] for c in offensive_labels]), axis=1)
        df['post'] = df['comment_text']

    if dataset_name == "jigsaw_bugged":

        df['post'] = df['bugged_post']

    if dataset_name == "jigsaw_buggable":

        df['post'] = df['orig_post']
        
        
    elif dataset_name == "convabuse":
        
        df['targets'] = df.apply(lambda x : _aggregate_convabuse_annotation(x), axis=1)
        #df['post'] = df.apply(lambda x : ' [SEP] '.join([x['prev_agent'], x['prev_user'], x['agent'], x['user']]), axis=1)
        df['post'] = df.apply(lambda x : '\n'.join([x['agent'], x['user']]), axis=1)

        
        
    elif dataset_name == "simsimi":
        
        #TODO

        df['targets'] = df.apply(lambda x : 1 if x['checkClassNormalProb']<=4 else 0, axis=1)
        df['post'] = df.apply(lambda x: x['reqSentence']+'\n'+x['respSentence'], axis=1)
        
        """
        corpus_request = df['reqSentence'].values
        corpus_response = df['respSentence'].values
        corpus_diag = df.apply(lambda x: x['reqSentence']+'\n'+x['respSentence'],axis=1).values
        targets = (df['checkClassNormalProb']<=4).astype(int).values
        
        targets = targets[:,0]
        target_corpus = corpus_diag
        testds = OffLangdataset(bert=args.bert,targets=targets,text_list=target_corpus)
        testloader = DataLoader(testds, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)
        """
        
        
        
    elif dataset_name == "sbic":
        
        #df['targets'] = df.apply(lambda x : 1 if x['offensiveYN'] > 0.5 else 0, axis=1)
        #df['targets'] = df.apply(lambda x : 1 if x['sexYN'] > 0.5 else 0, axis=1)
        df['targets'] = df.apply(lambda x : 1 if x['offensiveYN'] > 0 else 0, axis=1)

    elif dataset_name == "bbf":

        df['post'] = df['text']
       
    
    
    
    
def _balance_dataset(df):
    
    # sample and concat to create a balanced dataset
    
    toxic_df = df[df['targets'] == 1]
    nontoxic_df = df[df['targets'] != 1]
    
    if toxic_df.shape[0] < nontoxic_df.shape[0]:
    
        sampled_nontoxic_df = nontoxic_df.sample(toxic_df.shape[0])
        df = pd.concat([toxic_df, sampled_nontoxic_df])
        
    else:
        
        sampled_toxic_df = toxic_df.sample(nontoxic_df.shape[0])
        df = pd.concat([sampled_toxic_df, nontoxic_df])
    
    return df
    
        
        
        
def _aggregate_convabuse_annotation(x):
    
    template = "Annotator{i}_is_abuse.{s}"
    
    offensive = [-1, -2, -3]
    non_offensive = [0, 1]
    
    offensive_vote = 0
    
    for i in range(1,9):
        
        if '1' in [x[template.format(i=i, s=s)] for s in offensive]:
            offensive_vote += 1
        elif '1' in [x[template.format(i=i, s=s)] for s in non_offensive]:
            offensive_vote -= 1
            
    if offensive_vote > 0:
        return 1 # majority voted on 'offensive'
    else:
        return 0 # majority voted on 'non-offensive'
            
            

    
    
    
