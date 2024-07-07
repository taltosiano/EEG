import pandas as pd
from typing import Dict, List
import json

marker_data = pd.read_csv('./data/Cerebral Circles_EPOCX_S1_2021.07.10T21.48.59 11.00(1)_intervalMarker.csv')
eeg_data = pd.read_csv('./data/Cerebral Circles_EPOCX_S1_2021.07.10T21.48.59 11.00.md.mc.pm.fe.bp.csv', skiprows=1)

# delete prelimnaries
marker_data = marker_data[30:384]
eeg_data = eeg_data[6069:18649]
start_col = eeg_data.columns.get_loc('EEG.RawCq')
# Drop all columns from 'EEG.RawCq' to the end
columns_to_drop = eeg_data.columns[start_col:]
eeg_data = eeg_data.drop(columns=columns_to_drop)
eeg_data = eeg_data.drop(columns=['EEG.Counter', 'EEG.Interpolated'])

data = {} # Dict[samp: {eeg: 
i = 0
eeg_idx = 0
dict_idx = 0

while i < marker_data.shape[0]:
    if (marker_data.iloc[i]['type'] == 'keydown') \
            or (i < marker_data.shape[0] + 1 and
                    ((marker_data.iloc[i]['type'] == 'pattern' and marker_data.iloc[i + 1]['type'] == 'keydown') or
                    (marker_data.iloc[i]['type'] == 'plain_hit' and marker_data.iloc[i + 1]['type'] == 'gap_element')))\
            or (marker_data.iloc[i]['type'] != 'pattern' and
                marker_data.iloc[i]['type'] != 'plain_hit' and marker_data.iloc[i]['type'] != 'gap_element'):
        i += 1
        continue

    duration = marker_data.iloc[i]['duration']
    data[dict_idx] = {'eeg_dat':[],
                      'label': marker_data.iloc[i]['type']}
    if eeg_idx < eeg_data.shape[0]:
        start = eeg_data.iloc[eeg_idx, 0]
        while eeg_idx < eeg_data.shape[0] and eeg_data.iloc[eeg_idx, 0] < start + duration:
            data[dict_idx]['eeg_dat'].append(list(eeg_data.iloc[eeg_idx, 1:]))
            eeg_idx += 1

    dict_idx += 1
    i += 1

############ train, val, test split ###############
import random
from sklearn.model_selection import train_test_split

keys = list(data.keys())
random.shuffle(keys)

# Split ratios
train_ratio = 0.9
val_ratio = 0.05
test_ratio = 0.05

# Calculate split sizes
total_samples = len(keys)
train_size = int(total_samples * train_ratio)
val_size = int(total_samples * val_ratio)
test_size = total_samples - train_size - val_size

# Assign 'train', 'val', 'test' labels
for i, key in enumerate(keys):
    if i < train_size:
        data[key]['set'] = 'train'
    elif i < train_size + val_size:
        data[key]['set'] = 'val'
    else:
        data[key]['set'] = 'test'


# Save the dictionary to a JSON file
with open('./data/prep_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Data saved to prep_data.json")

############ split train,data,test #########
def del_pattern(data):
    new_dict = data
    for dict_idx, value in data.items():
        if value['label'] != 'pattern':
            new_dict[dict_idx] = value

    return new_dict

ignore_pattern = True
if ignore_pattern:
    del_pattern(data)

# Convert dictionary to list of items
data_list = [(key, value) for key, value in data.items()]

# Split the data into train, validation, and test sets
train_val, test = train_test_split(data_list, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1 , random_state=42)  # 0.1 x 0.9 = 0.09

# Convert back to dictionary while maintaining structure
train_dict = {i: {'eeg_dat': value['eeg_dat'], 'label': value['label']} for i, (key,value) in enumerate(train)}
val_dict = {i: {'eeg_dat': value['eeg_dat'], 'label': value['label']} for i, (key,value) in enumerate(val)}
test_dict = {i: {'eeg_dat': value['eeg_dat'], 'label': value['label']} for i, (key,value) in enumerate(test)}

# Save to JSON files
with open('./data/train.json', 'w') as f:
    json.dump(train_dict, f, indent=4)

with open('./data/val.json', 'w') as f:
    json.dump(val_dict, f, indent=4)

with open('./data/test.json', 'w') as f:
    json.dump(test_dict, f, indent=4)

print("Data has been split and saved to train.json, val.json, and test.json.")





