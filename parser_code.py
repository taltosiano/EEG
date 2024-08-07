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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# I/O args
parser.add_argument("--data-train", type=str, default='./data/train.json', help="training data json")
parser.add_argument("--data-val", type=str, default='./data/val.json', help="validation data json")
parser.add_argument("--data-eval", type=str, default='./data/test.json', help="evaluation data json")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
# training and optimization args
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=40, help="number of maximum training epochs")
parser.add_argument("--n-print-steps", type=int, default=1000, help="number of steps to print statistics")
# model args
parser.add_argument("--model", type=str, default="efficientnet", help="eeg model architecture", choices=["efficientnet", "svm"])
parser.add_argument("--eff_b", type=int, default=0, help="which efficientnet to use, the larger number, the more complex")
parser.add_argument("--n_class", type=int, default=2, help="number of classes")
parser.add_argument('--impretrain', help='if use imagenet pretrained CNNs', type=ast.literal_eval, default='True')
parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function", choices=["BCE", "CE"])
parser.add_argument("--lrscheduler_start", type=int, default=10, help="when to start decay")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay ratio")
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument("--warmup", help='if use balance sampling', type=ast.literal_eval, default='True')

parser.add_argument("--kernel", type=str, default="linear", help="kernel type to be used in the algorithm", choices=["linear", 'poly', 'rbf', 'sigmoid'])
parser.add_argument("--c", type=float, default=1.0, help="Regularization parameter")

args = parser.parse_args()

