import numpy as np
import base64
from tqdm import tqdm
import csv
import pickle
import joblib





TRAIN_PATH = 'data/train.sample.tsv'
VAL_PATH = 'data/valid.tsv'
VAL_ANS_PATH = 'data/valid_answer.json'
SAMPLE_PATH = 'data/train.sample.tsv'
LABEL_PATH = 'data/multimodal_labels.txt'
TESTA_PATH = 'data/testA.tsv'
TESTB_PATH = 'data/testB.tsv'

def get_label(path):
    with open(path) as f:
        lines = f.readlines()
        label2id = {l.split('\n')[0].split('\t')[1]:int(l.split('\n')[0].split('\t')[0]) for l in lines[1:]}
        id2label = {int(l.split('\n')[0].split('\t')[0]):l.split('\n')[0].split('\t')[1] for l in lines[1:]}
    return label2id, id2label

label2id, id2label = get_label(LABEL_PATH)


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




if __name__ == '__main__':
    
    from distributed import Client
    client = Client()
    import modin.pandas as pd

    # 读10000条训练数据    
    train = pd.read_csv(TRAIN_PATH,sep='\t', quoting=csv.QUOTE_NONE, error_bad_lines=False)

    train['words'] = train['query']
    train['label_words'] = train.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    train['boxes_convert'] = train.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    train['features'] = train.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    train['pos'] = train.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)



    data = train[['label_words', 'features', 'pos', 'words']] 
    data.to_pickle('data/sample_train_data.pkl')


    
    val = pd.read_csv(VAL_PATH,sep='\t')
    val['boxes_convert'] = val.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    val['feature_convert'] = val.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    val['labels_convert'] = val.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
    val['label_words'] = val.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    val['pos'] = val.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)

    del val['boxes'], val['features'], val['class_labels']
    val.to_pickle('data/val_data.pkl')            

    test = pd.read_csv(TESTA_PATH,sep='\t')
    test['boxes_convert'] = test.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    test['feature_convert'] = test.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    test['labels_convert'] = test.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
    test['label_words'] = test.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    test['pos'] = test.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)

    del test['boxes'], test['features'], test['class_labels']
    test.to_pickle('data/testA_data.pkl')


    test = pd.read_csv(TESTB_PATH,sep='\t')
    test['boxes_convert'] = test.apply(lambda x: convertBoxes(x['num_boxes'], x['boxes']), axis=1)
    test['feature_convert'] = test.apply(lambda x: convertFeature(x['num_boxes'], x['features']), axis=1)
    test['labels_convert'] = test.apply(lambda x: convertLabel(x['num_boxes'], x['class_labels']), axis=1)
    test['label_words'] = test.apply(lambda x: convertLabelWord(x['num_boxes'], x['class_labels']), axis=1)
    test['pos'] = test.apply(lambda x: convertPos(x['num_boxes'], x['boxes_convert'], x['image_h'], x['image_w']), axis=1)
    
    del test['boxes'], test['features'], test['class_labels']
    test.to_pickle('data/testB_data.pkl')
