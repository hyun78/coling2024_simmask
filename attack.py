"""from textattack.transformations import WordSwapQWERTY, WordSwapNeighboringCharacterSwap, WordSwapRandomCharacterDeletion, CompositeTransformation
from textattack.constraints.pre_transformation import MaxModificationRate, RepeatModification
from textattack.augmentation import Augmenter

constraints = [MaxModificationRate(max_rate=0.2), RepeatModification()]
transformation = CompositeTransformation([WordSwapQWERTY(), WordSwapNeighboringCharacterSwap(), WordSwapRandomCharacterDeletion()])

#TODO homoglyphs (o -> 0 etc.. seems not to be included in the package)
augmenter = Augmenter(transformation=transformation, constraints=constraints)

s = "hello my name is Jisu Hong, Nice to meet you!"

after = augmenter.augment(s)
print(after)
"""

#TODO HomoglyphSwap isn't very extensive

import torch
import torch.nn as nn
from dataset import load_dataframe, extract_post_targets
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018
from textattack.shared.attacked_text import AttackedText
from transformers import AutoModel, AutoTokenizer
from textattack.models.wrappers.pytorch_model_wrapper import PyTorchModelWrapper
from model import PredictionWithPretrained
from utils import parse_args
import os

class TokenizerWrapper:
    def __init__(self, tokenizer, max_len=3):
        self._tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, string_list):
        return self.tokenizer(string_list)

    def tokenizer(self, string_list):

        out = self._tokenizer(string_list, padding='max_length', truncation=True, 
                                       max_length=self.max_len, return_tensors='pt')
        out = torch.stack([out["input_ids"], out["attention_mask"]], dim=1)

        return out


class AttackedModelWrapper(nn.Module):

    def __init__(self, base_model:PredictionWithPretrained):

        super(AttackedModelWrapper, self).__init__()
        self.model = base_model

    def forward(self, input):
        #input: stacked tensor of input_ids and attention_mask from TokenizerWrapper
        input = torch.unbind(input, dim=1)

        input_ids = input[0]
        attention_mask = input[1]

        return self.model(input_ids, attention_mask)



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args=parse_args()

    model = PredictionWithPretrained(AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens'))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) # device_ids=device_ids
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    tokenizer_wrapper = TokenizerWrapper(tokenizer)

    best_rseed = [19, 10, 19, 16, 16]

    params = torch.load(f"model_path/{args.random_seed}_sbert_finetune_sbert_random_jigsaw_model_{best_rseed[args.random_seed]}.pth", map_location='cpu')

    model.load_state_dict(params, strict=True)

    wrapped_model = AttackedModelWrapper(model)

    model_wrapper = PyTorchModelWrapper(wrapped_model, tokenizer_wrapper)

    attack = TextBuggerLi2018.build(model_wrapper)

    print("ready")

    for i in range(10):
        x = AttackedText(input())
        print(attack.attack(x, 1))


