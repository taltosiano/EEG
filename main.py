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
parser.add_argument("--n-epochs", type=int, default=2, help="number of maximum training epochs")
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
# option to make audio conf with argparse.
# this will be in main.py
# audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm,
#               'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train',
#               'mean': args.dataset_mean, 'std': args.dataset_std,
#               'noise': False}
###################### DATA LOADING #######################################
train_loader = torch.utils.data.DataLoader(
        dataloader.EEGDataset(args.data_train),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    dataloader.EEGDataset(args.data_val),
    batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

eval_loader = torch.utils.data.DataLoader(
        dataloader.EEGDataset(args.data_eval),
        batch_size=args.batch_size*2, shuffle=False, num_workers=0, pin_memory=True)

############################ MODEL IMPORTING ####################################
if args.model == 'efficientnet':
    eeg_model = models.EffNetAttention(label_dim=args.n_class, b=args.eff_b, pretrain=args.impretrain, head_num=args.att_head)
elif args.model == 'svm':
    eeg_model = models.EEG_SVM_Classifier(kernel=args.kernel, C=args.c, gamma='scale')


## save experiment in a directory
if not bool(args.exp_dir):
    print("exp_dir not specified, automatically naming one...")
    args.exp_dir = "exp/%s/EEGModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
        time.strftime("%Y%m%d-%H%M%S"), args.model, args.optim,
        args.lr, args.n_epochs)

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

################ train the model #####################################
if args.model == 'efficientnet':
    train(eeg_model, train_loader, val_loader, args)
elif args.model == 'svm':
    eeg_model.fit(args.data_train)
    val_accuracy = eeg_model.evaluate(args.data_val)
    test_accuracy = eeg_model.evaluate(args.data_eval)

print('---------------Result Summary---------------')
if args.model == 'efficientnet':
    if args.data_eval != None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # evaluate best single model
        sd = torch.load(args.exp_dir + '/models/best_eeg_model.pth', map_location=device)
        if not isinstance(eeg_model, nn.DataParallel):
            eeg_model = nn.DataParallel(eeg_model)
        eeg_model.load_state_dict(sd)
        print('---------------evaluate best single model on the validation set---------------')
        stats, _ = validate(eeg_model, val_loader, args, 'best_single_valid_set')
        val_mAP = np.mean([stat['AP'] for stat in stats])
        val_mAUC = np.mean([stat['auc'] for stat in stats])
        print("mAP: {:.6f}".format(val_mAP))
        print("AUC: {:.6f}".format(val_mAUC))
        print('---------------evaluate best single model on the evaluation/test set---------------')
        stats, _ = validate(eeg_model, eval_loader, args, 'best_single_eval_set', eval_target=True)
        eval_mAP = np.mean([stat['AP'] for stat in stats])
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        print("mAP: {:.6f}".format(eval_mAP))
        print("AUC: {:.6f}".format(eval_mAUC))
        np.savetxt(args.exp_dir + '/best_single_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

elif args.model == 'svm':
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    np.savetxt(args.exp_dir + '/SVM_result.csv', [val_accuracy, test_accuracy])



