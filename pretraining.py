from utils import *
from model import *
from dataset import get_train_corpus_targets, OffLangdataset #load_train_dataset
from train import train_model, save_model
from contrastive import load_contrastive_dataset
from torch.utils.data import DataLoader
import os
import logging

def contrastive_pretraining(args):

    train_corpus, train_targets, val_corpus, val_targets = get_train_corpus_targets(args)

    trainds = OffLangdataset(bert = args.bert, targets = train_targets, text_list = train_corpus)
    valds = OffLangdataset(bert = args.bert, targets = val_targets, text_list = val_corpus)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)
    valloader = DataLoader(valds, batch_size=args.batch_size, shuffle=False, num_workers=1,drop_last=True)

    cont_trainloader = load_contrastive_dataset(args, train_corpus, train_targets=train_targets)

    apply_moco = "moco" in args.model

    if apply_moco:
        model = MoCoModel(num_cls=2, K=args.K)
    else:
        model = Model(bert=args.bert, num_cls=2)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) # device_ids=device_ids
    model.cuda()

    args.pretraining = True 

    if args.model != "bert_finetune":

        model, val_res = train_model(args, model, trainloader, valloader, cont_trainloader, validation=False)

    args.pretraining = False

    if apply_moco:
        return model.module.moco.encoder_q.bert
    else:
        return model.module.encoder


if __name__=="__main__":

    os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

    args = parse_args(default=False)

    logging.info("args: \n" + str(args))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    logging.info("START PRETRAINING")

    pretrained = contrastive_pretraining(args)

    save_model(args, pretrained, "pretrained")

    logging.info("END OF PRETRAINING")