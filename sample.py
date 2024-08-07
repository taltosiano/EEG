import argparse
import os
import dataloader
from models_dir import models
from traintest import train, validate
import numpy as np
import time
import torch
from torch import nn
import ast
import pickle
import parser_code
from parser_code import args
import json
import random
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# choose exp_dir
exp_dir = 'exp/20240708-135613/EEGModel-efficientnet_Optim-adam_LR-0.0005_Epochs-40'
parser.add_argument("--exp-dir", type=str, default=exp_dir, help="directory to dump experiments")
parser.add_argument('--own_sample', help='if use external sample', type=ast.literal_eval, default='False')
### if own_sample is true
parser.add_argument("--eeg_csv", type=str, default='', help="csv of eeg meas. must be with 14 channels! we wouldn't pre-process it!"
                                                            "like we did in data_prep.py")
parser.add_argument("--eeg_label", type=str, default="gap_element", help="label of eeg meas.", choices=["gap_element", "plain_hit"])
###
parser.add_argument('--own_idx', help='if use desired sample from test set', type=ast.literal_eval, default='False')
parser.add_argument("--samp_idx", type=int, default=2, help="if own-idx is true, then notify it")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samp_args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = {0: 'gap_element', 1: 'plain_hit'}

if samp_args.own_sample == False: #take a sample from test set
    with open('data/test.json', 'r') as fp:
        data_json = json.load(fp)
    if samp_args.own_idx:
        if samp_args.samp_idx < len(data_json):
            samp_idx = samp_args.samp_idx
        else: raise ValueError("An error occurred due to invalid index value")
    else:
        samp_idx = int(random.choice(list(data_json.keys())))
    sample = data_json[str(samp_idx)]
    print("The chosen Sample is", sample)
    print('----------------------------------------------------------------------------')
    GT_csv = pd.read_csv(samp_args.exp_dir + '/predictions/eval_target.csv', header=None)
    pred_csv = pd.read_csv(samp_args.exp_dir + '/predictions/predictions_' + 'best_single_eval_set.csv', header=None)
    sample_GT = GT_csv.iloc[samp_idx]
    sample_pred = pred_csv.iloc[samp_idx]/pred_csv.iloc[samp_idx].sum()
    print("sample's label is: ", label_dict[sample_GT.idxmax()])
    print(label_dict[sample_GT.idxmax()])
    print(f"model gave {sample_pred[0]*100:.2f}% for gap_element and {sample_pred[1]*100:.2f}% for plain_hit")

else: #user give an input
    eeg_data = pd.read_csv(samp_args.eeg_csv)
    eeg_data = eeg_data.values.tolist()
    label = samp_args.eeg_label
    sample = {'0': {'eeg_dat': eeg_data, 'label': label}}

    with open('./data/sample.json', 'w') as f:
         json.dump(sample, f, indent=4)

    #pre-processing
    samp_loader = torch.utils.data.DataLoader(
            dataloader.EEGDataset(dataset_json_file='./data/sample.json', exp_dir=exp_dir),
            batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    sd = torch.load(exp_dir + '/models/best_eeg_model.pth', map_location=device)
    eeg_model = models.EffNetAttention(label_dim=args.n_class, b=args.eff_b, pretrain=args.impretrain, head_num=args.att_head)
    if not isinstance(eeg_model, nn.DataParallel):
        eeg_model = nn.DataParallel(eeg_model)

    eeg_model.load_state_dict(sd)

    eeg_model = eeg_model.to(device)
    # switch to evaluate mode
    eeg_model.eval()

    with torch.no_grad():
        for i, (eeg_input, labels) in enumerate(samp_loader):
            eeg_input = eeg_input.to(device)
            # compute output
            eeg_output = eeg_model(eeg_input)
            eeg_output = eeg_output.to('cpu').detach()
            sample_GT = labels

        print("The chosen Sample is", sample['0']['eeg_dat'])
        print('----------------------------------------------------------------------------')
        sample_pred = eeg_output
        print("sample's label is: ", label_dict[torch.argmax(labels)])
        #print(label_dict[sample_GT.idxmax()])
        print(f"model gave {sample_pred[0] * 100:.2f}% for gap_element and {sample_pred[1] * 100:.2f}% for plain_hit")





