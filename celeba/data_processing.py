import os
import zipfile
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import shutil
import pickle

labels_path = 'celeba/list_attr_celeba.txt'
image_path = 'celeba/img_align_celeba/'
split_path = 'celeba/list_eval_partition.txt'

labels_df = pd.read_csv(labels_path)

label_dict = {}
for i in range(1, len(labels_df)):
    label_dict[labels_df['202599'][i].split()[0]] = [x for x in labels_df['202599'][i].split()[1:]]

label_df = pd.DataFrame(label_dict).T
label_df.columns = (labels_df['202599'][0]).split()
label_df.replace(['-1'], ['0'], inplace = True)

# generate train/val/test
files = glob(image_path + '*.jpg')

split_file = open(split_path, "r")
lines = split_file.readlines()

os.mkdir('celeba/tmp/')
for i in ['train', 'val', 'test']:
    os.mkdir(os.path.join('celeba/tmp/', i))

train_file_names = []
train_dict = {}
valid_file_names = []
valid_dict = {}
test_file_names = []
test_dict = {}
for i in tqdm(range(len(lines))):
    file_name, sp = lines[i].split()
    sp = sp.split('\n')[0]
    if sp == '0':
        labels = np.array(label_df[label_df.index==file_name])
        train_dict[file_name] = labels
        train_file_names.append(file_name)
        source_path = image_path + file_name
        shutil.copy2(source_path, os.path.join('celeba/tmp/train', file_name))
    elif sp == '1':
        labels = np.array(label_df[label_df.index==file_name])
        valid_dict[file_name] = labels
        valid_file_names.append(file_name)
        source_path = image_path + file_name
        shutil.copy2(source_path, os.path.join('celeba/tmp/val', file_name))
    else:
        labels = np.array(label_df[label_df.index==file_name])
        test_dict[file_name] = labels
        test_file_names.append(file_name)
        source_path = image_path + file_name
        shutil.copy2(source_path, os.path.join('celeba/tmp/test', file_name))

train_df = pd.DataFrame(train_dict.values())
train_df.index = train_file_names
train_df.columns = ['labels']

valid_df = pd.DataFrame(valid_dict.values())
valid_df.index = valid_file_names
valid_df.columns = ['labels']

test_df = pd.DataFrame(test_dict.values())
test_df.index = test_file_names
test_df.columns = ['labels']

df = {}
df['train'] = train_df
df['val'] = valid_df
df['test'] = test_df
with open('celeba/data_frame.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('data frame saved')
