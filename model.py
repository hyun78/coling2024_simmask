from transformers import AutoTokenizer, AutoModel
import torch.nn as nn, pandas as pd
import torch
import moco.builder

#FIXME: now that the training is divided into two phases,
#What happens to validation? (fixed epoch?)

class Model(nn.Module):
    
    def __init__(self, bert, num_cls):
        super(Model, self).__init__()
        self.dim = 768
        self.encoder = AutoModel.from_pretrained(bert)
        self.hidden = 100
        self.mlp_projection = nn.Sequential(nn.Linear(self.dim, self.hidden), 
                                             nn.ReLU(), 
                                             nn.Linear(self.hidden,self.hidden,bias=True))
        self.mlp_prediction = nn.Sequential(nn.Linear(self.dim, self.hidden), 
                                             nn.ReLU(), 
                                             nn.Linear(self.hidden, num_cls, bias=True))

    def forward(self, input_ids, attention_mask, ce=False): 
        
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        embedding = mean_pooling(output, attention_mask)

        if ce:
            return self.mlp_prediction(embedding)
        else:
            return self.mlp_projection(embedding)

class SBERTBaseEncoder(nn.Module):

    def __init__(self, bert="sentence-transformers/bert-base-nli-mean-tokens", num_classes=100, dim=768):

        #FIXME num_classes in the original code = hidden layer dim??

        super(SBERTBaseEncoder, self).__init__()
        self.dim = dim
        self.hidden = num_classes
        self.bert = AutoModel.from_pretrained(bert)
        self.mlp_projection = nn.Sequential(nn.Linear(self.dim, self.hidden), 
                                             nn.ReLU(), 
                                             nn.Linear(self.hidden,self.hidden,bias=True))
    

    def forward(self, x):
        """
        x: tensor w 2 columns
         1st col: seq of input ids, 2nd col: seq of attention mask
        """

        input_ids = x[:,0]
        attention_mask = x[:,1]
        
        embedding = self.get_embedding(input_ids, attention_mask)

        return self.mlp_projection(embedding)
    
    def get_embedding(self, input_ids, attention_mask):

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = mean_pooling(output, attention_mask)

        return embedding


class MoCoModel(nn.Module):

    def __init__(self, num_cls=2, dim=768, hidden=100, K=2048):

        super(MoCoModel, self).__init__()
        self.dim = dim
        self.hidden = hidden
        self.moco = moco.builder.MoCo(SBERTBaseEncoder, dim=hidden, K=K)

        self.mlp_prediction = nn.Sequential(nn.Linear(self.dim, self.hidden), 
                                             nn.ReLU(), 
                                             nn.Linear(self.hidden, num_cls, bias=True))

    def forward(self, input_ids, attention_mask, pos_input_ids=None, pos_attention_mask=None, ce=True):

        if ce:
            
            embedding = self.moco.encoder_q.get_embedding(input_ids, attention_mask)
            """
            
            model.moc.encoder_q.eval()
            'for input in dataset:
                    ...
            ce=True
            with torch.no_grad():
                embedding = self.moco.encoder_q.get_embedding(input_ids, attention_mask)
            embedding = embedding.detach()
            return self.mlp_prediction(embedding)

            5epoch pre-train (CL) SupCon / SimCSE / SimMask / SimBad
            cont_loss = ... 
            10 epoch fine-tune (CE)
            model.moc.encoder_q.eval() (finetune-freeze )


            """
            return self.mlp_prediction(embedding)

        else:

            in_q = torch.cat([input_ids.unsqueeze(dim=1), attention_mask.unsqueeze(dim=1)], dim=1)
            in_k = torch.cat([pos_input_ids.unsqueeze(dim=1), pos_attention_mask.unsqueeze(dim=1)], dim=1)

            return self.moco(in_q, in_k)

"""
class ContrastiveModel(nn.Module):

    # for the contrastive pretraining
    
    def __init__(self, bert, num_cls):
        super(ContrastiveModel, self).__init__()
        self.dim = 768
        self.encoder = AutoModel.from_pretrained(bert)
        self.hidden = 100
        self.mlp_projection = nn.Sequential(nn.Linear(self.dim, self.hidden), 
                                             nn.ReLU(), 
                                             nn.Linear(self.hidden,self.hidden,bias=True))
        self.mlp_prediction = nn.Sequential(nn.Linear(self.dim, self.hidden), 
                                             nn.ReLU(), 
                                             nn.Linear(self.hidden, num_cls, bias=True))

    def forward(self, input_ids, attention_mask, ce=False): 
        
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        embedding = mean_pooling(output, attention_mask)

        if ce:
            return self.mlp_prediction(embedding)
        else:
            return self.mlp_projection(embedding)
"""

class PredictionWithPretrained(nn.Module):

    def __init__(self, pretrained, num_cls=2, dim=768, hidden=100):

        super(PredictionWithPretrained, self).__init__()
        self.dim = dim
        self.hidden = hidden
        self.pretrained = pretrained
        self.pretrained.eval()
        self.mlp_prediction = nn.Sequential(nn.Linear(dim, hidden),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden, num_cls, bias=True))
        
    def forward(self, input_ids, attention_mask, ce=True):

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] 
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        with torch.no_grad():
            output = self.pretrained(input_ids, attention_mask)

        embedding = mean_pooling(output, attention_mask)

        return self.mlp_prediction(embedding.detach())