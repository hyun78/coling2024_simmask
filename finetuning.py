from utils import *
from model import *
from dataset import get_train_corpus_targets, OffLangdataset 
from train import train_model
from torch.utils.data import DataLoader
import os
import logging
from transformers import AutoModel
from pretraining import contrastive_pretraining

def get_pretrained_model(args):

    path = "model_path/"

    if args.others:

        pretrained_path = path + "{rseed}_{model}_sbert_{mode}_{dataset}_{others}_model_pretrained.pth".format(
            rseed=args.random_seed, model=args.model, mode=args.mode, dataset=args.train_dataset, others=args.others
        )

    else:

        pretrained_path = path + "{rseed}_{model}_sbert_{mode}_{dataset}_model_pretrained.pth".format(
            rseed=args.random_seed, model=args.model, mode=args.mode, dataset=args.train_dataset
        )

    if os.path.exists(pretrained_path):


        print("pretrained model exists: loading...")

        # pretrained model already exists: load that existing model

        params = torch.load(pretrained_path,map_location='cpu')

        """if "moco" in args.model:

            model = PredictionWithPretrained(MoCoModel(num_cls=2, K=args.K))
        
        else:

            model = PredictionWithPretrained(Model(bert=args.bert, num_cls=2))"""
        
        model = AutoModel.from_pretrained(args.bert)

        model.load_state_dict(params,strict=True)

        # if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model) # device_ids=device_ids
        # model.cuda()

        print("successfully loaded pretrained model")
    
    else: 

        print("pretrained model doesn't exist - start pretraining")
        
        # pretrained model doesn't exist: train a new one

        model = contrastive_pretraining(args)

        print("end of pretraining")

    return model


def finetune(args, pretrained_model):

    #can it be called finetuning? (encoder params are not updated...)

    model = PredictionWithPretrained(pretrained_model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) # device_ids=device_ids
    model.cuda()

    args.model = "{}_finetune".format(args.model)

    train_corpus, train_targets, val_corpus, val_targets = get_train_corpus_targets(args)

    trainds = OffLangdataset(bert = args.bert, targets = train_targets, text_list = train_corpus)
    valds = OffLangdataset(bert = args.bert, targets = val_targets, text_list = val_corpus)

    trainloader = DataLoader(trainds, batch_size=args.batch_size, shuffle=True, num_workers=1,drop_last=True)
    valloader = DataLoader(valds, batch_size=args.batch_size, shuffle=False, num_workers=1,drop_last=True)

    model, val_res = train_model(args, model, trainloader, valloader, validation=True)

    return model, val_res


if __name__=="__main__":

    os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

    args = parse_args(default=False)

    logging.info("args: \n" + str(args))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    logging.info("START PRETRAINING (or load existing pretrained model)")

    if args.model == "bert_finetune":
        pretrained = AutoModel.from_pretrained(args.bert)
        args.model = "sbert"

    else:
        pretrained = get_pretrained_model(args)

    logging.info("START FINETUNING")

    finetuned, val_res = finetune(args, pretrained)

    logging.info("validation result: \n" + str(val_res))

    logging.info("END")

    