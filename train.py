from torch.optim import Adam,SGD
import torch.nn as nn
import torch.nn.functional as F
from utils import InfoNCE, get_res
from utils import AverageMeter
from tqdm import tqdm
import torch
from supcon.losses import SupConLoss


def train_model(args, model, trainloader, valloader, cont_trainloader=None, validation=True):

    optimizer = Adam(model.parameters(), lr=args.lr)

    val_epoch_res = []
    
    for epoch in range(args.num_epochs):
        
        model.train()
        
        update_params(args, model, optimizer, trainloader, valloader, cont_trainloader)
        
        if validation:
            model.eval()

            with torch.no_grad():
                validation_result = get_res(model, valloader)
            
            val_epoch_res.append(validation_result)
            save_model(args, model, epoch)
    
    if validation:
        save_val(args, val_epoch_res)
    
    return model, val_epoch_res





def update_params(args, model, optimizer, trainloader, valloader, cont_trainloader=None):
    
    model_name = args.model
    
    losses = AverageMeter()
    tbar = tqdm(trainloader)
    
    if cont_trainloader != None:
        
        # contrastive learning: involves contrastive loss
        
        contbar = iter(cont_trainloader)
              
        
    for inputs, targets, _ in tbar:
        
        if cont_trainloader != None:
            
            # contrastive learning (Simmask, SimCSE etc.)
            
            try:
                org_inputs, pos_inputs = next(contbar)
                
            except StopIteration:
                contbar = iter(cont_trainloader)
                org_inputs, pos_inputs = next(contbar)
                
            loss = get_loss(args, model, inputs, targets, org_inputs, pos_inputs) # 얘를 loss list에 append해서 모니터링 (저장해서 visualization) x: it, y: loss
                
        else:
            
            loss = get_loss(args, model, inputs, targets)
                 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), args.batch_size)
        tbar.set_description("loss: {}".format(losses.avg), refresh=True)
        
        
    
def get_loss(args, model, inputs, targets, org_inputs=None, pos_inputs=None):
    
    model_name = args.model
    
    if "finetune" in model_name:
        
        return get_ce_loss(args, model, inputs, targets)
    
    elif "sim" in model_name:

        #if args.pretraining == True:
            #return get_cont_loss(args, model, org_inputs, pos_inputs, targets=targets)

        #else:
        return get_ce_loss(args, model, inputs, targets) + args.lambda_ * get_cont_loss(args, model, org_inputs, pos_inputs, targets=targets)
    
    
    elif model_name == "tfidf_svm":
        pass
        
    
    else:
        raise

        
    
def get_cont_loss(args, model, org_inputs, pos_inputs, targets=None):

    org_input_ids = org_inputs['input_ids'].long().cuda()
    pos_input_ids = pos_inputs['input_ids'].long().cuda()

    org_attention_mask = org_inputs['attention_mask'].long().cuda()
    pos_attention_mask = pos_inputs['attention_mask'].long().cuda()

    if "moco" in args.model:
        # moco outputs logits computed from cosine similarity of embeddings
        cont_criterion = nn.CrossEntropyLoss()

        logits, labels = model(input_ids=org_input_ids, attention_mask=org_attention_mask, 
                               pos_input_ids=pos_input_ids, pos_attention_mask=pos_attention_mask, ce=False)
        
        cont_loss = cont_criterion(logits, labels)

    elif "supcon" in args.model:

        cont_criterion = SupConLoss(temperature=args.temperature, base_temperature=args.temperature)

        org_embedding = model(input_ids=org_input_ids, attention_mask=org_attention_mask, ce=False).cuda()
        pos_embedding = model(input_ids=pos_input_ids, attention_mask=pos_attention_mask, ce=False).cuda()
        all_embeddings = torch.stack([org_embedding, pos_embedding], dim=1)
        # print("shape:", all_embeddings.shape) # [16, 2, 100]
        all_embeddings = F.normalize(all_embeddings, dim=2, p=2) #

        cont_loss = cont_criterion(all_embeddings, targets)

    else:
    
        cont_criterion = InfoNCE(temperature=args.temperature, batch_size=args.batch_size)
        
        
        cont_input_ids = torch.cat([org_input_ids,pos_input_ids]).cuda()
        cont_attention_mask = torch.cat([org_attention_mask,pos_attention_mask]).cuda()
        cont_output = model(input_ids=cont_input_ids, attention_mask=cont_attention_mask, ce=False)

        cont_loss = cont_criterion(cont_output)

        del cont_input_ids, cont_attention_mask, cont_output
    
    return cont_loss


def get_ce_loss(args, model, inputs, targets):

    ce_criterion = nn.CrossEntropyLoss()

    
    input_ids = inputs['input_ids'].long().cuda()
    attention_mask = inputs['attention_mask'].long().cuda()
    targets = targets.long().cuda()

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    celoss = ce_criterion(output,targets)

    del targets, input_ids, output, attention_mask
    
    return celoss


def save_model(args, model, epoch):


    if args.others==None:
    
        model_save_path = args.model_path + f'/{args.random_seed}_{args.model}_{args.bert_name}_{args.mode}_{args.train_dataset}_model_{epoch}.pth'

    else:

        model_save_path = args.model_path + f'/{args.random_seed}_{args.model}_{args.bert_name}_{args.mode}_{args.train_dataset}_{args.others}_model_{epoch}.pth'

    torch.save(model.state_dict(), model_save_path)
    

def save_val(args, val):

    if args.others==None:
    
        eval_save_path = args.eval_path + f'/{args.random_seed}_{args.model}_{args.bert_name}_{args.mode}_{args.train_dataset}_{args.test_dataset}_val.pth'
    
    else:
    
        eval_save_path = args.eval_path + f'/{args.random_seed}_{args.model}_{args.bert_name}_{args.mode}_{args.train_dataset}_{args.test_dataset}_{args.others}_val.pth'
    
    torch.save(val, eval_save_path)
