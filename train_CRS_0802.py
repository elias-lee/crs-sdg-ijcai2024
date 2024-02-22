import numpy as np
import pandas as pd
import torch
import math
import time
import datetime
import pdb
import os
import re
import gc
import fileinput
import string
#import tensorflow as tf
import sys
from tqdm  import tqdm
tqdm.pandas()

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AdamW #ElectraForSequenceClassification, AutoTokenizer,
from tqdm.notebook import tqdm
import pickle
#from multilabelmodel import ElectraForMultiLabelClassificationSoho
from transformers import ElectraModel, ElectraTokenizer
from sklearn.metrics import top_k_accuracy_score

# from nltk.tokenize import wordpunct_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
import pdb

# from keras.preprocessing.text import Tokenizer
# # from keras.preprocessing.sequence import pad_sequences
# from keras.utils import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# # from keras.layers.embeddings import Embedding
# from tensorflow.keras.layers import Embedding
from sklearn.metrics import classification_report
from torch_ema import ExponentialMovingAverage


dtlist= ["/SSD/rudeh6185/CRS_data_journal/CRS_data_1973-2021_/CRS_2018_data.txt", "/SSD/rudeh6185/CRS_data_journal/CRS_data_1973-2021_/CRS_2019_data.txt", "/SSD/rudeh6185/CRS_data_journal/CRS_data_1973-2021_/CRS_2020_data.txt","/SSD/rudeh6185/CRS_data_journal/CRS_data_1973-2021_/CRS_2021_data.txt"]
dflist=[]
for dtname in dtlist:
  temp = pd.read_csv(dtname, sep="|", encoding = "ISO-8859-1")
  dflist.append(temp)


df_temp = pd.concat(dflist)
df_temp = df_temp.reset_index(drop=True)

df_clean = df_temp.loc[:,["DonorCode","SDGfocus","LongDescription","ProjectTitle", "ShortDescription",
                    "AgencyCode", "RecipientCode","RegionCode","IncomegroupCode","FlowCode", "Bi_Multi"]]
df_clean = df_clean[df_clean['LongDescription'].str.len() > 25].copy()

# Dropna before cleaning
print("Dimension before droping na was {number1}x{number2} ".format(number1=str(df_clean.shape[0]), number2=str(df_clean.shape[1])))
df_clean = df_clean.dropna(subset=['SDGfocus'])
print("Dimension after droping na was {number1}x{number2} ".format(number1=str(df_clean.shape[0]), number2=str(df_clean.shape[1])))
#pdb.set_trace()

df_clean['LongDescription'] = df_clean['LongDescription'].astype(str)


cou_dict = {#"Australia": 801,
            #"Belgium":2,
            "Canada":301,
            "Denmark":3,
            "EU":918,
            "Finland":18,
            "France":4,
            "Germany":5,
            "Iceland":20,
            "Ireland":21,
            "Italy":6,
            "Japan":701,
            "Korea":742,
            "Luxembourg":22,
            "Netherlands":7,
            #"New Zealand":820,
            "Norway":8,
            #"Sweden":10,
            #"Switzerland":11,
            #"United Kingdom":12,
            #"United States":302
            }

# cou_dict = {"Australia": 801,
#             "Belgium":2,
#             "Canada":301,
#             "Denmark":3,
#             "EU":918,
#             "Finland":18,
#             "France":4,
#             "Germany":5,
#             "Iceland":20,
#             "Ireland":21,
#             "Italy":6,
#             "Japan":701,
#             "Korea":742,
#             "Luxembourg":22,
#             "Netherlands":7,
#             "New Zealand":820,
#             "Norway":8,
#             "Sweden":10,
#             "Switzerland":11,
#             "United Kingdom":12,
#             "United States":302
#             }



# pdb.set_trace()
donorcodes_to_keep = list(cou_dict.values())
# Filter the DataFrame based on the "donorcode" column
print("Shape BEFORE cleaning donors"+ str(df_clean.shape))
print('-' * 50)
filtered_df = df_clean[df_clean['DonorCode'].isin(donorcodes_to_keep)]
print("Shape AFTER cleaning donors"+ str(filtered_df.shape))

new_df=filtered_df[["SDGfocus", "LongDescription"]]
new_df['SDGfocus']= new_df['SDGfocus'].astype(str)
# Step 1: Clean the labels by eliminating everything that comes after a period(.)
def clean_label(label):
    label = label.split('.')[0].strip()
    label = label.split('b')[0].strip()
    label = label.split('d')[0].strip()
    return label
new_df['SDGfocus'] = new_df['SDGfocus'].apply(lambda x: ';'.join([clean_label(label) for label in x.split(';')]))
new_df['SDGfocus'] = new_df['SDGfocus'].str.replace(',', ';')
new_df['SDGfocus'] = new_df['SDGfocus'].str.replace(' ', '')
# result_df = pd.concat([filtered_df['SDGfocus'], test['SDGfocus']], axis=1, ignore_index=True)
# result_df.to_csv('merged_sdg_focus.csv', index=False)
#---------------------------------------------------------#
# Step 2: Eliminate duplicates within each observation
new_df['SDGfocus'] = new_df['SDGfocus'].apply(lambda x: ';'.join(list(set(x.split(';')))))
# Step 3: Remove semicolons at the end without any values after them
new_df['SDGfocus'] = new_df['SDGfocus'].apply(lambda x: x.rstrip(';') if x.endswith(';') else x)
new_df['SDGfocus'] = new_df['SDGfocus'].apply(lambda x: x.lstrip(';') if x.startswith(';') else x)
new_df['SDGfocus'] = new_df['SDGfocus'].str.replace(';;', ';')
#---------------------------------------------------------#
labels = set(label for sublist in new_df['SDGfocus'].str.split(';') for label in sublist)
for label in labels:
    new_df[label] = new_df['SDGfocus'].apply(lambda x: 1 if label in x.split(';') else 0)
# pd.DataFrame(filtered_df.iloc[24627]).transpose()

print("Shape BEFORE column 0"+ str(new_df.shape))
print('-' * 50)
#pdb.set_trace()
if '0' in new_df: 
    new_df = new_df[new_df['0'] != 1].drop(columns=['0'])
print("Shape AFTER column 0"+ str(new_df.shape))
new_df.head(2)



import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)
def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

# Usage
replace_contractions("this's a text with contraction")
# lower the text
new_df["LongDescription"] = new_df["LongDescription"].apply(lambda x: x.lower())

# Clean the text
new_df["LongDescription"] = new_df["LongDescription"].apply(lambda x: clean_text(x))

# Clean numbers
new_df["LongDescription"] = new_df["LongDescription"].apply(lambda x: clean_numbers(x))

# Clean Contractions
new_df["LongDescription"] = new_df["LongDescription"].apply(lambda x: replace_contractions(x))

def column_types(df):
    for i in df.columns:
        class_name = str(type(df[i][0]))
        print("column "+ i + " is " + class_name)

        new_df.reset_index(drop=True)



new_df.drop(labels=['SDGfocus'], axis=1, inplace=True)
print(new_df.shape)
new_df = new_df.dropna()
print(new_df.shape)

import torch.nn as nn

# Shuffle the rows randomly using sample with frac=1 (frac=1 means using all rows)
randomized_df = new_df.sample(frac=1, random_state=42)

# Calculate the indices where to split the DataFrame
split_index1 = int(len(randomized_df) * 6/10)  # 1/3 of the rows
split_index2 = int(len(randomized_df) * 8/10)  # 2/3 of the rows

# Split the DataFrame into three separate DataFrames
train_df = randomized_df.iloc[:split_index1]
test_df = randomized_df.iloc[split_index1:split_index2]
val_df = randomized_df.iloc[split_index2:]

# Reset the index of the split DataFrames (optional, if needed)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

target_list = ['1','2','3','4','5','6','7','8','9',
               '10','11','12','13','14','15','16','17']

MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 100
#LEARNING_RATE = 1e-04
LEARNING_RATE = 1e-05

from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['LongDescription']
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }


train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import shutil

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)



class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 17)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

model = BERTClass()
model.to(device)

def accuracy_fc(y_pred,y_true, count):
    #val_outputs
    #val_targets
    #37958 total
    #pdb.set_trace()
    

    for a in range(0, len(y_true)):



        
        # y_arg = np.argmax(y_pred[a])
        # y_arg = int(y_arg)
        y_arg_1 = int(np.argsort(y_pred[a])[-1])
        y_arg_2 = int(np.argsort(y_pred[a])[-2])
        y_arg_3 = int(np.argsort(y_pred[a])[-3])
        y_arg_4 = int(np.argsort(y_pred[a])[-4])
        y_arg_5 = int(np.argsort(y_pred[a])[-5])

        #pdb.set_trace()
        if y_true[a].count(1) == 1:
            if int(y_true[a][y_arg_1])==1:

                count = count +1

        # elif int(y_true[a][y_arg_2]) ==1 or int(y_true[a][y_arg_3]) ==1: #or int(y_true[a][y_arg_4]) ==1 or int(y_true[a][y_arg_5]) ==1:#top-3
        elif int(y_true[a][y_arg_2]) ==1 or int(y_true[a][y_arg_3]) ==1: or int(y_true[a][y_arg_4]) ==1 or int(y_true[a][y_arg_5]) ==1:#top-5

            count = count +1
        # if int(y_true[a][y_arg_1])==1:
        # if int(y_true[a][y_arg_2]) ==1 or int(y_true[a][y_arg_3]) ==1 or int(y_true[a][y_arg_4]) ==1 or int(y_true[a][y_arg_5]) ==1:
            #if int(y_true[a][y_arg_2]) ==1 or int(y_true[a][y_arg_3]) ==1:

                # count = count +1
        # if y_true[a].count(1.0) == 1 and np.argmax(y_pred[a]) ==  np.argmax(y_true[a]):

        #     count = count + 1

        # #elif y_true[a].count(1.0) >1 and np.argsort(y_true[a])[-y_true[a].count(1.0):] == np.argsort(y_pred[a])[-y_true[a].count(1.0):]:
        # elif (np.argsort(y_true[a])[-2:]).tolist() == (np.argsort(y_pred[a])[-2:]).tolist():
        #     count = count + 1



    return count
           
def EMA(net):
    ema = EMA(
        net,
        beta = 0.9999,              # exponential moving average factor
        update_after_step = 100,    # only after this number of .update() calls will it start updating
        update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    # mutate your network, with SGD or otherwise

    with torch.no_grad():
        net.weight.copy_(torch.randn_like(net.weight))
        net.bias.copy_(torch.randn_like(net.bias))

    # you will call the update function on your moving average wrapper

    ema.update()

    # then, later on, you can invoke the EMA model the same way as your network

    return ema

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def MSE_loss_fn(outputs, targets):
    return torch.nn.MSELoss()(outputs, targets)
    
    # class_weight = torch.tensor([ 1.0329,  1.4150,  1.0077,  0.8853,  1.1958,  2.6353,  5.2724,  1.5071,
    #      2.0611,  1.8790,  3.2205,  4.3517,  2.0317, 18.8524,  3.4310,  0.8791,
    #      1.6601]).cuda()
    # loss = torch.nn.MultiLabelSoftMarginLoss(weight=class_weight, reduction='mean')
    # return loss(outputs, targets)
#BCEWithLogitsLoss
#MultiLabelSoftMarginLoss

# def loss_fn_class_weight(outputs, targets):
#     class_weight = ([ 1.0329,  1.4150,  1.0077,  0.8853,  1.1958,  2.6353,  5.2724,  1.5071,
#          2.0611,  1.8790,  3.2205,  4.3517,  2.0317, 18.8524,  3.4310,  0.8791,
#          1.6601])
#     loss = torch.nn.BCEWithLogitsLoss(weight=class_weight, reduction='mean')
#     return loss(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


val_targets=[]
val_outputs=[]

val_targets_softmax=[]
val_outputs_softmax=[]







def train_model(n_epochs, training_loader, validation_loader, model,
                optimizer):

  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf


  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0
    first_class= torch.zeros(1,17).cuda()
    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        #print('yyy epoch', batch_idx)
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        # ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
        # with ema.average_parameters():
        #     outputs_EMA= ema(ids)
        #     MSE_loss = MSE_loss_fn(outputs, outputs_EMA)

           

        
        # ema.update()
        
        target_num = torch.sum(targets, dim=0)

        first_class = target_num + first_class
        loss = loss_fn(outputs, targets)


        # loss = loss + MSE_loss

        #loss = loss_fn_class_weight(outputs, targets, class_weight)
        #if batch_idx%5000==0:
         #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)

    
    print('target_num',first_class)   
    print('############# Epoch {}: Training End     #############'.format(epoch))

    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################
    # validate the model #
    ######################

    model.eval()
    test_class= torch.zeros(1,17).cuda()
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)

            

            
            target_num = torch.sum(targets, dim=0)

            test_class = target_num + test_class
            
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_targets_softmax.extend(targets.argmax(axis=1).cpu().detach().numpy().tolist())
            #pdb.set_trace()
            #.softmax(dim=1)
            #targets.argmax(axis=1)
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            output_softmax = outputs.softmax(dim=1)
            val_outputs_softmax.extend(output_softmax.argmax(axis=1).cpu().detach().numpy().tolist())


      print('test_num',test_class)   
      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics
      #pdb.set_trace()
      count = 0
      acc_count = accuracy_fc(y_pred=val_outputs,y_true=val_targets, count=count)
      acc = sklearn.metrics.accuracy_score(y_pred=val_outputs_softmax,y_true=val_targets_softmax)
      f1 = sklearn.metrics.f1_score(y_true=val_targets_softmax,y_pred=val_outputs_softmax,average='macro')
      #auc = sklearn.metrics.roc_auc_score(y_true=val_targets_softmax,y_score=val_outputs_softmax,multi_class='ovr')
      prec = sklearn.metrics.precision_score(y_true=val_targets_softmax,y_pred=val_outputs_softmax,average='macro')
      recall = sklearn.metrics.recall_score(y_true=val_targets_softmax,y_pred=val_outputs_softmax,average='macro')

      print("real_acc", acc_count/37958/int(epoch))
      #print("real_acc2", acc_count/37958)
      print("acc:", acc)
      print("f1_score:", f1)
      print("prec:", prec)
      print("recall:", recall)
      count = 0

      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
            ))

      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
      #pdb.set_trace()

      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss,valid_loss))

        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model


trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer)

