from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import pandas as pd
import os
import random
import sklearn


def read_data():
    all_data = pd.read_csv(os.path.join("..","data","test_data.csv"))
    all_text = all_data["content"].tolist()
    return all_text

def resplit_text(text_list):
    result = []
    sentence = ""
    for text in text_list:
        if len(text) < 3:
            continue
        if sentence == "":
            if random.random()<0.2:
                result.append(text + "。")
                continue

        if len(sentence) < 30 or random.random()<0.2:
            sentence += text + "，"
        else:
            result.append(sentence[:-1] + "。")
            sentence = text

    return result

def split_text(text):
    # patten = r"。|？"
    patten = r"[，、：；。？]"
    sp_text = re.split(patten,text)
    new_sp_text  = resplit_text(sp_text)
    return new_sp_text

def build_neg_pos_data(text_list):
    all_text1,all_text2 = [],[]
    all_label = []


    for tidx , text in enumerate(text_list):
        if tidx == len(text_list)-1:
            break
        all_text1.append(text)
        all_text2.append(text_list[tidx+1])
        all_label.append(1)

        c_id = [i for i in range(len(text_list)) if i != tidx and i != tidx+1]


        other_idx = random.choice(c_id)

        other_text = text_list[other_idx]
        all_text1.append(text)
        all_text2.append(other_text)
        all_label.append(0)

    return all_text1,all_text2,all_label


def build_task2_dataset(text_list):
    all_text1 = []
    all_text2 = []
    all_label = []

    for text in tqdm(text_list):
        sp_text = split_text(text)
        if len(sp_text)<=2:
            continue
        text1,text2,label = build_neg_pos_data(sp_text)

        all_text1.extend(text1)
        all_text2.extend(text2)
        all_label.extend(label)

    pd.DataFrame({"text1":all_text1,"text2":all_text2,"label":all_label}).to_csv(os.path.join("..","data","task2.csv"),index=False)


def build_word_2_index(all_text):
    if os.path.exists("index_2_word.txt") == True:
        with open("index_2_word.txt",encoding="utf-8") as f:
            index_2_word = f.read().split("\n")
            word_2_index = {w:idx for idx,w in enumerate(index_2_word)}
            return word_2_index,index_2_word
    word_2_index = {"[PAD]":0,"[unused1]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"[UNK]":5,}

    for text in all_text:
        for w in text:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
    index_2_word = list(word_2_index)

    with open("index_2_word.txt","w",encoding="utf-8") as f:
        f.write("\n".join(index_2_word))


    return word_2_index,index_2_word


def get_data():
    all_data = pd.read_csv(os.path.join("..","data","task2.csv"))

    # all_data = sklearn.utils.shuffle(all_data)

    text1 = all_data["text1"].tolist()
    text2 = all_data["text2"].tolist()
    label = all_data["label"].tolist()

    return text1,text2,label


class BDataset(Dataset):
    def __init__(self,all_text1,all_text2,all_lable,max_len,word_2_index):
        assert len(all_text1) == len(all_text2) == len(all_lable),"数据长度都不一样，复现个冒险啊！"
        self.all_text1 = all_text1
        self.all_text2 = all_text2
        self.all_lable = all_lable
        self.max_len = max_len
        self.word_2_index = word_2_index


    def __getitem__(self, index):
        text1 = self.all_text1[index]
        text2 = self.all_text2[index]

        lable = self.all_lable[index]

        text1_idx = [word_2_index.get(i,self.word_2_index["[UNK]"]) for i in text1][:62]
        text2_idx = [word_2_index.get(i,self.word_2_index["[UNK]"]) for i in text2][:62]



        mask_val = [0] * self.max_len

        text_idx = [self.word_2_index["[CLS]"]] + text1_idx + [self.word_2_index["[SEP]"]] + text2_idx + [self.word_2_index["[SEP]"]]
        seg_idx = [0] + [0] * len(text1_idx) + [0] + [1] * len(text2_idx) + [1] + [2] * (self.max_len - len(text_idx))

        for i,v in enumerate(text_idx):
            if v in [self.word_2_index["[CLS]"],self.word_2_index["[SEP]"],self.word_2_index["[UNK]"]] :
                continue

            if random.random() < 0.15:
                r = random.random()
                if  r < 0.8:
                    text_idx[i] = self.word_2_index["[MASK]"]

                    mask_val[i] = v

                elif r > 0.9:
                    other_idx = random.randint(6,len(self.word_2_index)-1)
                    text_idx[i] = other_idx
                    mask_val[i] = v


        text_idx = text_idx + [self.word_2_index["[PAD]"] ]* (self.max_len - len(text_idx))


        return torch.tensor(text_idx) , torch.tensor(lable) ,torch.tensor(mask_val),torch.tensor(seg_idx)



    def __len__(self):
        return len(self.all_lable)


class BertEmbeddding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.word_embeddings.weight.requires_grad = True

        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.position_embeddings.weight.requires_grad = True

        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.token_type_embeddings.weight.requires_grad = True

        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])


    def forward(self,batch_idx,batch_seg_idx):
        word_emb = self.word_embeddings(batch_idx)

        pos_idx = torch.arange(0,self.position_embeddings.weight.data.shape[0])
        pos_idx = pos_idx.repeat(10,1)
        pos_emb = self.position_embeddings(pos_idx)

        token_emb = self.token_type_embeddings(batch_seg_idx)

        emb = word_emb + pos_emb + token_emb

        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        return emb


class BertPooler(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]

        return first_token_tensor


class BertModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embedding = BertEmbeddding(config)
        self.bert_layer = nn.Linear(config["hidden_size"],config["hidden_size"])
        self.pool = BertPooler()

    def forward(self,batch_idx,batch_seg_idx):
        emb = self.embedding(batch_idx,batch_seg_idx)
        bertout1 =  self.bert_layer(emb)
        bertout2 = self.pool(bertout1)

        return bertout1,bertout2


class Model(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.bert = BertModel(config) #  3*128*768 @  768 * 768 = 3*128*768 @ 768 * 10 = 3 * 128 * 10,             3*768  @ 768 * 2  = 3 * 2

        self.cls_mask = nn.Linear(config["hidden_size"],config["vocab_size"])
        self.cls_next = nn.Linear(config["hidden_size"],2)


    def forward(self,batch_idx,batch_seg_idx):
        bert_out = self.bert(batch_idx,batch_seg_idx)
        print("")




if __name__ == "__main__":
     # all_text = read_data()
     # build_task2_dataset(all_text)
     # word_2_index = build_word_2_index(all_text)

     all_text1,all_text2,all_label = get_data()

     with open("index_2_word.txt",encoding="utf-8") as f:
         index_2_word = f.read().split("\n")
         word_2_index = {w:idx for idx,w in enumerate(index_2_word)}

     epoch = 10
     batch_size = 10
     max_len = 128       #

     config = {
         "epoch" :epoch,
         "batch_size":batch_size,
         "max_len" : max_len,
         "vocab_size" : len(word_2_index),
         "hidden_size":768,
         "max_position_embeddings":128,
         "type_vocab_size":3,
         "hidden_dropout_prob":0.2
     }





     train_dataset = BDataset(all_text1,all_text2,all_label,max_len,word_2_index)
     train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

     model = Model(config)

     for e in range(epoch):
         print(f"epoch : {e}")
         for batch_idx,batch_label,batch_mask_val,batch_seg_idx in train_dataloader:

             model.forward(batch_idx,batch_seg_idx)

     print("")

