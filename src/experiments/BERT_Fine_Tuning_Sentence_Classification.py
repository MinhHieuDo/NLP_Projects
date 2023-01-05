import pandas as pd
import os
import torch
import numpy as np
from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cpu')
project_dir  = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir  = project_dir + '/data/'
print('data_dir',data_dir)
df = pd.read_csv(data_dir+'in_domain_train.tsv',delimiter='\t',header=None,names=['sentence_source','label','label_notes','sentence'])
print(df.head())
sentences = df['sentence'].values
sentences = ["[CLS]" + sentence + " [SEP] " for sentence in sentences]
labels = df['label'].values
print(sentences[0])

## Activate BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]
print(tokenized_texts[0])
input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]
print(input_ids[0])
# mytokenized_texts = [tokenizer(sentence) for sentence in sentences]
# print(mytokenized_texts[0])

# Pad our input token
MAX_LEN = 128
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
print(input_ids[0])
#  Create a mask of 1s for each token followed by 0s for padding
attention_masks = [[float(id>0) for id in seq] for seq in input_ids]
print(attention_masks[0])
# Split data into train and validation set
X_train,X_val ,y_train,y_val=train_test_split(input_ids,labels,random_state=2022,test_size=0.1)
mask_train, mask_val ,_,_ = train_test_split(attention_masks,input_ids,random_state=2022,test_size=0.1)
# Transform data into tensor
X_train = torch.tensor(X_train,device=device)
X_val  = torch.tensor(X_val,device=device)
y_train = torch.tensor(y_train,device=device)
y_val = torch.tensor(y_val,device=device)
mask_train = torch.tensor(mask_train,device=device)
mask_val   = torch.tensor(mask_val,device=device)

# Create an iterator of our data with torch DataLoader: save memory during the training step
BATCH_SIZE = 64
train_data = TensorDataset(X_train,mask_train,y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,batch_size=BATCH_SIZE,sampler=train_sampler)

val_data = TensorDataset(X_val,mask_val,y_val)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE,sampler=val_sampler)

print(train_sampler)
configuration = BertConfig()
print(configuration)
# Initializing a model from the bert-base-uncased style configuration
#model = BertModel(configuration)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
model.to(device)
print(model)
# print(np.sum([np.prod(list(p.shape)) for p in model.parameters()]), 'parameters in BERT')
# for p in model.parameters():
#     print('p',p.shape)


## optimization 

EPOCHS = 2
# Create greoup parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    # filter all parameter w/o bias
    {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate':0.1},
    # w/o decay parameter
    {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate':0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,lr = 2e-5,eps = 1e-8)
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = len(train_dataloader)*EPOCHS)

# optimizer = torch.optim.Adam(params=model.parameters(),lr=1.e-5,betas=(0.8,0.95))
# scheduler = LinearLR(optimizer,start_factor=0.5,total_iters=5)

## Train model
train_loss_set = []
for epoch in tqdm(range(EPOCHS)):
    model.train()

    train_loss = 0
    nb_train_steps = 0
    nb_train_examples = 0

    for step, batch in enumerate(train_dataloader):
        # Add GPU
        batch = tuple(b.to(device) for b in batch)
        X_train,mask_train,y_train = batch
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train,token_type_ids=None,attention_mask=mask_train, labels=y_train)
        loss = outputs['loss']
        train_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
        train_loss += loss.item()
        nb_train_steps += 1
        nb_train_examples = X_train.size(0)
        print('Step = {}, loss = {}'.format(step,loss.item()))
    print('Train loss = {}'.format(train_loss/nb_train_steps))
    
    # Validation
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0
    for batch in (val_dataloader):
        batch = tuple(b.to(device) for b in batch)
        X_val,mask_val,y_val = batch
        with torch.no_grad():
            outputs = model(X_val,token_type_ids=None,attention_mask=mask_val, labels=y_val)
        logits = outputs['logits'].detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1).flatten()
        targets = y_val.cpu().numpy()
        accuracy = accuracy_score(targets,predictions)

        eval_accuracy += accuracy
        nb_eval_steps  += 1
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    

plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()






