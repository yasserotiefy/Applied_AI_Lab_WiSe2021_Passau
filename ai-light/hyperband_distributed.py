#!/usr/bin/env python
# coding: utf-8

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import json
import pandas as pd
import numpy as np
from random import choice, seed, shuffle, random, sample
import tensorflow as tf
# import tensorflow.co as KTF
from keras.models import Sequential, Model
from keras.layers import Input, CuDNNGRU as GRU, CuDNNLSTM as LSTM, Dropout, BatchNormalization
from keras.layers import Dense, Concatenate, Activation, Embedding, SpatialDropout1D, Bidirectional, Lambda, Conv1D
from keras.layers import Add, Average, TimeDistributed, GlobalMaxPooling1D
from tensorflow.compat.v1.keras.optimizers import Adam, Nadam
# from keras.activations import absolute_import
# from keras.legacy import interfaces
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
# from keras.utils import to_categorical
import tensorflow.compat.v1.keras.backend as K
import keras_tuner as kt
# from keras.callbacks import ModelCheckpoint
import keras
# from sklearn.model_selection import KFold
# from keras.initializers import he_normal
# from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert import AdamWarmup, calc_train_steps
from keras import activations, initializers, regularizers, constraints
from keras.models import Model
# from tqdm import tqdm
# from model_utils import seq_gather, seq_and_vec, seq_maxpool
from keras.models import load_model
from keras_bert import get_custom_objects
from keras_bert import Tokenizer
from collections import defaultdict
from eval import read_submission, get_ndcg
from tqdm import tqdm, trange
import pickle
import dask.dataframe as dd


BERT_PRETRAINED_DIR = "/root/Applied_AI_Lab_WiSe2021_Passau/ai-light/data/uncased_L-12_H-768_A-12"
VAL_ANS_PATH = '/root/Applied_AI_Lab_WiSe2021_Passau/ai-light/data/valid_answer.json'
LABEL_PATH = '/root/Applied_AI_Lab_WiSe2021_Passau/ai-light/data/multimodal_labels.txt'

MAX_EPOCH = 20
MAX_LEN = 10
B_SIZE = 64
FOLD_IDS = [-1]
FOLD_NUM = 20
THRE = 0.5
SHUFFLE = True
MAX_BOX = 5
MAX_CHAR = 5
PREFIX = "[image-bert-concat-query]-wwm_uncased_L12-768_v3_1M_example"
SEED = 2021
ACCUM_STEP = int(128 // B_SIZE)
SAVE_EPOCHS=[10, 20, 35, 50, 80, 100]
IMAGE_LABEM_CONCAT_TOKEN = "###"
CONCAT_TOKE = "[unused0]"

cfg = {}
cfg["verbose"] = PREFIX
cfg["base_dir"] = BERT_PRETRAINED_DIR
cfg['maxlen'] = MAX_LEN
cfg["max_box"] = MAX_BOX
cfg["max_char"] = MAX_CHAR
cfg["lr"] = 1e-4
cfg['min_lr'] = 6e-8
cfg["opt"] = "nadam"
cfg["loss_w"] =  20.
cfg["trainable"] = True
cfg["bert_trainable"] = True
cfg["mix_mode"] = ""   # add concat average
cfg["unit1_1"] = 128
cfg["accum_step"] = ACCUM_STEP
cfg["cls_num"] = 2
cfg["raw_filename"] = "{}_{}oof{}"



def get_vocab():
    
    if "albert"in cfg["verbose"].lower():
        dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab_chinese.txt')
    else:
        dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    with open(dict_path, mode="r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    word_index = {v: k  for k, v in enumerate(lines)}
    return word_index


word_index = get_vocab()
cfg["x_pad"] = word_index["[PAD]"]
tokenizer = Tokenizer(word_index)


def get_label(path):
    with open(path) as f:
        lines = f.readlines()
        label2id = {l.split('\n')[0].split('\t')[1]:int(l.split('\n')[0].split('\t')[0]) for l in lines[1:]}
        id2label = {int(l.split('\n')[0].split('\t')[0]):l.split('\n')[0].split('\t')[1] for l in lines[1:]}
    return label2id, id2label


label2id, id2label = get_label(LABEL_PATH)
label_set = set(label2id.keys())


# In[3]:


import joblib
# 全量数据
# with open('../data/train_data.pkl', 'rb') as outp:
#     train_data= joblib.load(outp)
# 100K sample
with open('data/sample_train_data.pkl', 'rb') as outp:
    train_data = joblib.load(outp)


with open('data/val_data.pkl', 'rb') as outp:
    val_data = pickle.load(outp)
    


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype=np.float16)


def load_embed(path, dim=300, word_index=None):
    embedding_index = {}
    with open(path, mode="r", encoding="utf8") as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().split()
            word, arr = l[0], l[1:]
            if len(arr) != dim:
                print("[!] l = {}".format(l))
                continue
            if word_index and word not in word_index:
                continue
            word, arr = get_coefs(word, arr)
            embedding_index[word] = arr
    return embedding_index


def build_matrix(path, word_index=None, max_features=None, dim=300):
    embedding_index = load_embed(path, dim=dim, word_index=word_index)
    max_features = len(word_index) + 1 if max_features is None else max_features 
    embedding_matrix = np.zeros((max_features + 1, dim))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
    return embedding_matrix, unknown_words


def load_word_embed(word_embed_glove="data/glove.840B.300d.txt", 
                    word_embed_crawl="data/crawl-300d-2M.vec",
               save_filename="word_embedding_matrix",
               word_index=None):
    """
    (30524, 300) 7590
    (30524, 300) 7218
    """    
    if os.path.exists(save_filename + ".npy"):
        word_embedding_matrix = np.load(save_filename + ".npy").astype("float32")
    else:
        word_embedding_matrix, tx_unk = build_matrix(word_embed_glove, word_index=word_index, dim=300)

        # print(word_embedding_matrix.shape, len(tx_unk))
        
        word_embedding_matrix_v2, tx_unk = build_matrix(word_embed_crawl, word_index=word_index, dim=300)

        # print(word_embedding_matrix_v2.shape, len(tx_unk))
        
        word_embedding_matrix = np.concatenate([word_embedding_matrix, word_embedding_matrix_v2], axis=1)
        
        gc.collect()
        np.save(save_filename, word_embedding_matrix)
    return word_embedding_matrix


word_embedding_matrix = load_word_embed(word_index=word_index)


# In[5]:



# In[6]:


import csv
import base64


def convertBoxes(num_boxes, boxes):
    return np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
def convertFeature(num_boxes, features,):
    return np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
def convertLabel(num_boxes, label):
    return np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
def convertLabelWord(num_boxes, label):
    temp = np.frombuffer(base64.b64decode(label), dtype=np.int64).reshape(num_boxes)
    return '###'.join([id2label[t] for t in temp])
def convertPos(num_boxes, boxes, H, W):
    pos_list = []
    for i in range(num_boxes):
        temp = boxes[i]
        pos_list.append([temp[0]/W, 
                         temp[2]/W, 
                         temp[1]/H, 
                         temp[3]/H, 
                         ((temp[2] - temp[0]) * (temp[3] - temp[1]))/ (W*H),])
    return pos_list


def load_sample_data(path, frac=0.000001):
    ### Data Sampling and decoding
    train_data = dd.read_csv(path, sep='\t', blocksize=25e6, quoting=csv.QUOTE_NONE, error_bad_lines=False)
    
    sample_train_data = train_data.sample(frac=frac).compute()
    sample_train_data['words'] = sample_train_data['query']
    sample_train_data['label_words'] = sample_train_data.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1,meta= pd.Series([], dtype=str, name='label_words'))
    sample_train_data['boxes_convert'] = sample_train_data.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1, meta =pd.Series([], dtype=object, name='boxes_convert'))
    sample_train_data['features'] = sample_train_data.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1, meta=pd.Series([], dtype=object, name='features'))
    sample_train_data['pos'] = sample_train_data.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1, meta=pd.Series([], dtype=object, name='pos'))
    return sample_train_data[['words', 'label_words', 'boxes_convert', 'features', 'pos']].compute()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            print(len(tokens_a))
            tokens_b.pop()


def token2id_X(X, x_dict, maxlen=None):
    x = tokenizer.tokenize(X)
    if maxlen:
        x = x[: 1] + list(x)[1: maxlen - 1] + x[-1: ]     
    seg = [0 for _ in x]
    token = list(x)
    x = [x_dict[e] if e in x_dict else x_dict["[UNK]"] for e in token]
    assert len(x) == len(seg)
    return x, seg


def seq_padding(X, maxlen=None, padding_value=None, debug=False):
    L = [len(x) for x in X]
    if maxlen is None:
        maxlen = max(L)

    pad_X = np.array([
        np.concatenate([x, [padding_value] * (maxlen - len(x))]) if len(x) < maxlen else x[: maxlen] for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X
    

def MyChoice(Myset):
    result = []
    for i in Myset:
        temp_set = set()
        temp_set.add(i)
        cho = choice(list(Myset - temp_set))
        result.append(cho)
    return result


class data_generator:
    
    def __init__(self, data, batch_size=B_SIZE, shuffle=SHUFFLE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.shuffle = shuffle

        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps
    

    def __iter__(self):
        """
        inp_token1,
        inp_segm1,
        inp_image,
        inp_image_mask,
        inp_pos, 
        inp_image_char
        """
        

        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            T1, T2, Image1, Pos1, label_word_list, image1_mask, image1_char = [], [], [], [], [], [], []
            S1, S2, Image2, Pos2, image2_mask, image2_char = [], [], [], [], [], [] # 负样本
            Id_set = set()

            for i in idxs:
                d = self.data.iloc[i]
                text = d['words']
                label_words = d['label_words']
                
                t1, t2 = token2id_X(text, x_dict=word_index, maxlen=cfg["maxlen"])
                image = np.array(d['features'], dtype="float32")
                image = image[: cfg["max_box"]]
                img_mask = [1 for _ in image[: cfg["max_box"]]]
                
                pos = np.array(d['pos'], dtype="float32")
                pos = pos[: cfg["max_box"]]
                
                image_char = [token2id_X(ent, x_dict=word_index)[0] for ent in label_words.split(IMAGE_LABEM_CONCAT_TOKEN)]
                image_char = image_char[: cfg["max_box"]]
                # print("image_char", len(image_char))
                image_char = pad_sequences(image_char, 
                                           maxlen=cfg["max_char"], 
                                           dtype='int32',
                                           padding='post',
                                           truncating='post',
                                           value=cfg["x_pad"])
                
                assert image.shape[0] == pos.shape[0]
                assert image.shape[0] == cfg["max_box"] or image.shape[0] == len(label_words.split(IMAGE_LABEM_CONCAT_TOKEN))
                assert image_char.shape == (image.shape[0], cfg["max_char"])

                T1.append(t1)
                T2.append(t2)
                Image1.append(image)
                image1_mask.append(img_mask)  
                Pos1.append(pos)
                image1_char.append(image_char)
                Id_set.add(i)

                if len(T1) == self.batch_size//2 or i == idxs[-1]:
                    ## 加入负样本
                    Id_new = MyChoice(Id_set)
#                     print(Id_set, Id_new)
                    for i, id_ in enumerate(Id_new):
                        d_new = self.data.iloc[id_]
                        text = d_new['words']
                        t1, t2 = token2id_X(text, x_dict=word_index, maxlen=cfg["maxlen"])
                        S1.append(t1)
                        S2.append(t2)
                        
                        image = Image1[i]
                        img_mask = image1_mask[i]
                        pos = Pos1[i]
                        image_char = image1_char[i]
                        
                        Image2.append(image)
                        Pos2.append(pos)
                        image2_mask.append(img_mask)
                        image2_char.append(image_char)
                    
                    Y = [1] * len(T1) + [0] * len(S1)
                   
                    T1 = seq_padding(T1 + S1, padding_value=cfg["x_pad"]) 
                    T2 = seq_padding(T2 + S2, padding_value=cfg["x_pad"])
                    
                    Image1 = seq_padding(Image1 + Image2, 
                                         padding_value=np.zeros(shape=(2048, ))
                                        )
                                                         
                    Pos1 = seq_padding(Pos1 + Pos2,
                                       padding_value=np.zeros(shape=(5, ))
                                      )
                    image1_mask = seq_padding(image1_mask + image2_mask,
                                             padding_value=0)
                    
                    image1_char = seq_padding(image1_char + image2_char,
                                             padding_value=np.zeros(shape=(cfg["max_char"])), debug=False)
                    
                    Y = np.array(Y).reshape((len(T1), -1))
                    
                    idx = np.arange(len(T1))
                    np.random.shuffle(idx)
        
                    T1 = T1[idx]
                    T2 = T2[idx]
                    Image1 = Image1[idx]
                    image1_mask = image1_mask[idx]
                    Pos1 = Pos1[idx]
                    image1_char = image1_char[idx]
                    Y = Y[idx]
                    
                    yield [T1, T2, Image1, image1_mask, Pos1, image1_char], Y
                    T1, T2, Image1, Pos1, label_word_list, image1_mask, image1_char = [], [], [], [], [], [], []
                    S1, S2, Image2, Pos2, image2_mask, image2_char = [], [], [], [], [], [] # 负样本
                    Id_set = set()

                        


# In[7]:


# import pandas as pd
# TRAIN_PATH = "data/train.tsv"
# data = load_sample_data(TRAIN_PATH, frac=0.00001)

# print("data", len(data))

# train_data = pd.read_pickle("data/100K_data.pkl")

train_D = data_generator(train_data)
with open("data/sample_val.pkl", "rb") as f:
    val_D = pickle.load(f)
val_D = data_generator(val_data)


class Evaluate(Callback):
    def __init__(self, filename=None):
        self.score = []
        self.best = 0.
        self.filename = filename
       
    def on_epoch_begin(self, epoch, logs=None):
        if epoch ==  0:
            print("[!] test load&save model")
            f = self.filename + ".h5"
            custom_objects = get_custom_objects()
            self.model.save(f, include_optimizer=False, overwrite=True)
            if "bert" in cfg["verbose"]:
                model_ = load_model(f, custom_objects=custom_objects)  
            else:
                model_ = load_model(f) 
    
    def on_epoch_end(self, epoch, logs=None):
#         if epoch + 1 < 5:
#             return
        score = self.evaluate(self.model)
        self.score.append((epoch, score))
        
        if epoch + 1 in SAVE_EPOCHS:
            self.model.save(self.filename + "_{}.h5".format(epoch + 1), include_optimizer=False, overwrite=True)             
        if score > self.best:
            self.model.save(self.filename + ".h5", include_optimizer=False)
            
        if score > self.best:
            self.best = score
            print("[!] epoch = {}, new NDCG best score = {}".format(epoch + 1,  score))
        print('[!] epoch = {}, score = {}, NDCG best score: {}\n'.format(epoch + 1, score, self.best))

    def eval_preprocess(self, row):

            d = row
            # qid = d['query_id']
            # pid = d['product_id']
            text = d['query']
            label_words = d['label_words']
            t1, t2 = token2id_X(text, x_dict=word_index, maxlen=cfg["maxlen"])
            
            image = np.array(d['feature_convert'], dtype="float32")
            image = image[: cfg["max_box"]]
            img_mask = [1 for _ in image[: cfg["max_box"]]]                   
            pos = np.array(d['pos'], dtype="float32")
            pos = pos[: cfg["max_box"]]
            
            image_char = [token2id_X(ent, x_dict=word_index)[0] for ent in label_words.split(IMAGE_LABEM_CONCAT_TOKEN)]
            image_char = image_char[: cfg["max_box"]]
            image_char = pad_sequences(image_char, 
                                       maxlen=cfg["max_char"], 
                                       dtype='int32',
                                       padding='post',
                                       truncating='post',
                                       value=cfg["x_pad"])
            output = self.model.predict([np.asarray([t1]), np.asarray([t2]), np.asarray([image]), np.asarray([img_mask]), np.asarray([pos]), np.asarray([image_char])])
            return output


    def evaluate(self, model):
        self.model = model
        result = defaultdict(list)
        val_results = val_data.apply(self.eval_preprocess, axis=1)
        # print(val_results)
        qid = val_data["query_id"].values
        pid = val_data["product_id"].values


        # print(val.shape)
        

        for i in trange(len(val_data)): 
            result[qid[i]].append((pid[i], val_results[i][0][1]))
            
        query_id,product1,product2,product3,product4,product5 = [],[],[],[],[],[]
        for key in result.keys():
            rlist = result[key]
            rlist.sort(key=lambda x: x[1], reverse=True)
            query_id.append(key)
            product1.append(rlist[0][0])
            product2.append(rlist[1][0])
            product3.append(rlist[2][0])
            product4.append(rlist[3][0])
            product5.append(rlist[4][0])
        sub = pd.DataFrame({'query-id':query_id,
                            'product1':product1,
                            'product2':product2,
                            'product3':product3,
                            'product4':product4,
                            'product5':product5,

        })
        sub.to_csv('result/val_submission.csv',index=0)
        
        reference = json.load(open(VAL_ANS_PATH))
        
        # read predictions
        k = 5
        predictions = read_submission('result/val_submission.csv', reference, k)

        # compute score for each query
        score_sum = 0.
        for qid in reference.keys():
            ground_truth_ids = set([str(pid) for pid in reference[qid]])
            ref_vec = [1.0] * len(ground_truth_ids)
            pred_vec = [1.0 if pid in ground_truth_ids else 0.0 for pid in predictions[qid]]
            score_sum += get_ndcg(pred_vec, ref_vec, k)
        # the higher score, the better
        score = score_sum / len(reference)
        
        return score


# ## Hyperparameter Optimization


# In[9]:





# In[10]:


def build_model(hp):

    global cfg
    
    def _get_model(base_dir, cfg_=None):
        config_file = os.path.join(base_dir, 'bert_config.json')
        checkpoint_file = os.path.join(base_dir, 'bert_model.ckpt')
        if not os.path.exists(config_file):
            config_file = os.path.join(base_dir, 'bert_config_large.json')
            checkpoint_file = os.path.join(base_dir, 'roberta_l24_large_model')
        print(config_file, checkpoint_file)
#         model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=cfg_['maxlen'])
        model = load_trained_model_from_checkpoint(config_file, 
                                           checkpoint_file, 
                                           training=False, 
                                           trainable=cfg_["bert_trainable"], 
                                           output_layer_num=cfg["cls_num"],
                                           seq_len=None)
        return model
    
    def get_opt(num_example, warmup_proportion=0.1, lr=2e-5, min_lr=None):
        if cfg["opt"].lower() == "nadam":
            opt = Nadam(lr=lr)
        else:
            total_steps, warmup_steps = calc_train_steps(
                num_example=num_example,
                batch_size=B_SIZE,
                epochs=MAX_EPOCH,
                warmup_proportion=warmup_proportion,
            )

            opt = AdamWarmup(total_steps, warmup_steps, lr=lr, min_lr=min_lr)

        return opt

    # model1 = _get_model(cfg["base_dir"], cfg)
    # model1 = Model(inputs=model1.inputs[: 2], outputs=model1.layers[-7].output)

    global word_index
    word_embedding_matrix = load_word_embed(word_index=word_index)
    embed_layer = Embedding(input_dim=word_embedding_matrix.shape[0], 
                            output_dim=word_embedding_matrix.shape[1],
                            weights=[word_embedding_matrix],
                            trainable=cfg["trainable"],
                            name="embed_layer"
                        )
        
    inp_token1 = Input(shape=(None, ), dtype=np.int32, name="query_token_input")
    inp_segm1 = Input(shape=(None, ), dtype=np.float32, name="query_segm_input")
    
#     inp_token2 = Input(shape=(None, ), dtype=np.int32)
#     inp_segm2 = Input(shape=(None, ), dtype=np.float32)    
    
    inp_image = Input(shape=(None, 2048), dtype=np.float32, name="image_input")
    inp_image_mask = Input(shape=(None, ), dtype=np.float32, name="image_mask_input")
    inp_pos = Input(shape=(None, 5), dtype=np.float32, name="image_pos_input")        
    inp_image_char = Input(shape=(None, cfg["max_char"]), dtype=np.int32, name='image_char_input')
    
    
    mask = Lambda(lambda x: K.cast(K.not_equal(x, cfg["x_pad"]), 'float32'), name="token_mask")(inp_token1)
    word_embed = embed_layer(inp_token1)
    word_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([word_embed, mask])
    
    hp_units_lstm = hp.Int('units', min_value=64, max_value=512, step=32)
    word_embed = Bidirectional(LSTM(hp_units_lstm, return_sequences=True), merge_mode="sum")(word_embed)
    word_embed = BatchNormalization()(word_embed)
    # word_embed = Dropout(0.3)(word_embed)
    word_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([word_embed, mask])

    # sequence_output = model1([inp_token1, inp_segm1])
    # sequence_output = Concatenate(axis=-1)([sequence_output, word_embed])
    text_pool = Lambda(lambda x: x[:, 0, :])(word_embed)

    # Share weights of character-level embedding for premise and hypothesis
    hp_units_filter = hp.Int('units', min_value=64, max_value=512, step=32)
    hp_units_filter_size = hp.Int('units', min_value=3, max_value=12, step=2)
    character_embedding_layer = TimeDistributed(Sequential([
        embed_layer,
        # Embedding(input_dim=100, output_dim=char_embedding_size, input_length=chars_per_word),
        Conv1D(filters=hp_units_filter, kernel_size=hp_units_filter_size, padding='same', name="char_embed_conv1d"),
        GlobalMaxPooling1D()
    ]), name='CharEmbedding')
    character_embedding_layer.build(input_shape=(None, None, cfg["max_char"]))
    image_char_embed  = character_embedding_layer(inp_image_char)    
    image_embed = Concatenate(axis=-1)([image_char_embed, inp_image])
    hp_units0 = hp.Int('units', min_value=64, max_value=2048, step=32)
    image_embed = Dense(hp_units0, name='image_embed')(image_embed)
    image_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([image_embed, inp_image_mask])


    hp_units = hp.Int('units', min_value=64, max_value=2048, step=32)
    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'selu', 'elu'])
    pos_embed = Dense(hp_units, activation=hp_activation, name='pos_embed')(inp_pos)
    pos_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([pos_embed, inp_image_mask])
    embed = Add()([image_embed , pos_embed]) # batch, maxlen(10), 1024+128
    
    hp_units_lstm0 = hp.Int('units', min_value=64, max_value=512, step=32)
    image_embed = Bidirectional(LSTM(hp_units_lstm0, return_sequences=True), merge_mode="sum")(embed)
    image_embed = BatchNormalization()(image_embed)
    image_embed = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([image_embed, inp_image_mask])
    
    image_pool = Lambda(lambda x: x[:, 0, :])(image_embed)
    
    pool = Concatenate(axis=-1)([image_pool, text_pool])

    hp_units = hp.Int('units', min_value=64, max_value=2048, step=32)
    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'selu', 'elu'])
    hp_dropout_prop = hp.Choice('dropout_prop', values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    pool = Dense(hp_units, activation=hp_activation)(pool)
    pool = Dropout(hp_dropout_prop)(pool)
    hp_units1 = hp.Int('units', min_value=64, max_value=2048, step=32)
    hp_activation1 = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'selu', 'elu'])
    pool = Dense(hp_units1, activation=hp_activation1)(pool)
    hp_units2 = hp.Int('units', min_value=64, max_value=2048, step=32)
    hp_activation2 = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid', 'selu', 'elu'])
    pool = Dense(hp_units2, activation=hp_activation2)(pool)
    
    output = Dense(2, activation='softmax', name='output')(pool)

    model = Model(inputs=[inp_token1, inp_segm1, 
                          inp_image, inp_image_mask,
                          inp_pos, inp_image_char], outputs=[output])#
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss={
                'output': 'sparse_categorical_crossentropy'
            }, metrics=['accuracy'])

    
    return model



tuner = kt.Hyperband(build_model,
                     objective=kt.Objective("val_accuracy", direction="max"),
                     max_epochs=20,
                     factor=3,
                     directory='tmp/hyperband_hyperparameter_tuning',
                     project_name='multimodal_hyperband_hyperparameter_tuning',
                     overwrite=True,
                     distribution_strategy=tf.compat.v1.distribute.experimental.MultiWorkerMirroredStrategy())

# Wrap data in Dataset objects.
output_signature=(
         tf.TensorSpec(shape=(None, None), dtype=tf.float16),
         tf.TensorSpec(shape=(None, None), dtype=tf.float16),
         tf.TensorSpec(shape=(5, 2048), dtype=tf.float16),
         tf.TensorSpec(shape=(5), dtype=tf.float16),
         tf.TensorSpec(shape=(None, 5), dtype=tf.float16),
         tf.TensorSpec(shape=(None, 5), dtype=tf.float16),
         tf.TensorSpec(shape=(), dtype=tf.float16))
train_data = tf.data.Dataset.from_generator(train_D.__iter__, output_signature=output_signature)
val_data = tf.data.Dataset.from_generator(val_D.__iter__, output_signature=output_signature)

# The batch size must now be set on the Dataset objects.
batch_size = 32
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options)
val_data = val_data.with_options(options)


stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
log_dir = "tmp/hparam_hyperband_logs"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

tuner.search(train_data, epochs=20, validation_data=val_data, callbacks=[stop_early, tensorboard_callback],
            # batch_size=64, steps_per_epoch=len(train_D)//4, validation_steps=len(val_D)//4
            )


