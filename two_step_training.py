import random
import numpy as np

import logging
from utils import *
from model import *
from dataset import get_train_corpus_targets, OffLangdataset #load_train_dataset
from train import train_model
from contrastive import load_contrastive_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2,3'

    args = parse_args(default=False)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    train_corpus, train_targets, val_corpus, val_targets = get_train_corpus_targets(args)
    
    trainds = OffLangdataset(bert = args.bert, targets = train_targets, text_list = train_corpus)
    valds = OffLangdataset(bert = args.bert, targets = val_targets, text_list = val_corpus)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)
    valloader = DataLoader(valds, batch_size=args.batch_size, shuffle=False, num_workers=1,drop_last=True)

    if "sim" in args.model:

        # contrastive pre-training

        cont_trainloader = load_contrastive_dataset(args, train_corpus, train_targets=train_targets)
        

        if "moco" in args.model:
            model = MoCoModel(num_cls=2, K=args.K)
        else:
            model = Model(bert=args.bert,num_cls = 2)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model) # device_ids=device_ids
        model.cuda()

        args.pretraining = True #FIXME

        model, val_res = train_model(args, model,trainloader,valloader,cont_trainloader, validation=False)

        if "moco" in args.model:
            model = PredictionWithPretrained(model.module.moco.encoder_q.bert, num_cls=2)
        else:
            model = PredictionWithPretrained(model.module.encoder)

        args.model = "{}_finetune".format(args.model)
    
    else:
         
        model = Model(bert=args.bert, num_cls=2)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) # device_ids=device_ids
    model.cuda()

    args.pretraining = False

    cont_trainloader = None

    model, val_res = train_model(args, model,trainloader,valloader,cont_trainloader)



    #trainloader,  valloader, cont_trainloader = load_train_dataset(args)

    ########


    torch.manual_seed(args.random_seed)

    if "moco" in args.model:
        model = MoCoModel(num_cls=2, K=args.K)
    else:
        model = Model(bert=args.bert,num_cls = 2)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) # device_ids=device_ids
    model.cuda()

    model, val_res = train_model(args, model,trainloader,valloader,cont_trainloader)
    logging.info("validation result: \n" + str(val_res))