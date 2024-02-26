import random
import numpy as np

from pretraining import contrastive_pretraining
from finetuning import finetune
from utils import parse_args
import logging
import os

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']='2,3'

    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')

    args = parse_args(default=False)

    logging.info("args: \n" + str(args))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    logging.info("START PRETRAINING")

    pretrained = contrastive_pretraining(args)

    logging.info("END OF PRETRAINING, START FINETUNING")

    finetuned, val_res = finetune(args, pretrained)
    
    logging.info("END OF FINETUNING")

    logging.info("validation result\n" + str(val_res))