import torch
import textattack
from transformers import AutoModel, AutoTokenizer
from utils import parse_args
from model import PredictionWithPretrained, Model
from dataset import load_test_dataset
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018


import os
os.environ['CUDA_VISIBLE_DEVICES']='2,3'


args = parse_args(default=False)

model = torch.nn.DataParallel(PredictionWithPretrained(AutoModel.from_pretrained(args.bert)), device_ids=[0,1])


best_rseed = [19, 10, 19, 16, 16]

params = torch.load(f"model_path/{args.random_seed}_sbert_finetune_sbert_random_jigsaw_model_{best_rseed[args.random_seed]}.pth", map_location='cpu')

model.load_state_dict(params, strict=True)

model = model.module
tokenizer = AutoTokenizer.from_pretrained(args.bert)

wrapper = textattack.models.wrappers.pytorch_model_wrapper.PyTorchModelWrapper(model, tokenizer)

attack = TextBuggerLi2018.build(wrapper)

print("hello world!")


#model_wrapper = PyTorchModelWrapper(model)


#attack = TextBuggerLi2018.build()

#new_corpus = []