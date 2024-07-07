import sys
import os
import datetime
from utilities import util
#  allows you to import modules from parent directory
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
import time
import torch
from torch import nn
import numpy as np
import pickle
'''
Mixed precision training involves using 16-bit floating point (float16) for most operations
while maintaining 32-bit (float32) precision for some critical parts of the model (like the loss computation) to preserve model accuracy. 
'''
from torch.cuda.amp import autocast,GradScaler

AverageMeter = util.AverageMeter
def train(eeg_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []

    # best_ensemble_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_ensemble_epoch, best_mAP, best_acc, best_ensemble_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # for GPU
    if not isinstance(eeg_model, nn.DataParallel):
        eeg_model = nn.DataParallel(eeg_model)

    eeg_model = eeg_model.to(device)
    # Set up the optimizer and count model parameters
    trainables = [p for p in eeg_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in eeg_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, 5)), gamma=args.lrscheduler_decay, last_epoch=epoch - 1)
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCELoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    args.loss_fn = loss_fn
    print('now training, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} '.format(args.lrscheduler_start, args.lrscheduler_decay))

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 7]) # saving mAP, AUC, recall..etc 10 stats
    eeg_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        eeg_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (eeg_input, labels) in enumerate(train_loader):

            B = eeg_input.size(0)
            eeg_input = eeg_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / eeg_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 100 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 100) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            eeg_output = eeg_model(eeg_input)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(eeg_output, torch.argmax(labels.long(), axis=1))
            else:
                epsilon = 1e-7
                eeg_output = torch.clamp(eeg_output, epsilon, 1. - epsilon)
                loss = loss_fn(eeg_output, labels)

            # optimization if amp is not used
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/eeg_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/eeg_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(eeg_model, test_loader, args, epoch)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions']) / 2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls']) / 2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        if main_metrics == 'mAP':
            result[epoch - 1, :] = [mAP, mAUC, average_precision, average_recall, loss_meter.avg,
                                    valid_loss, optimizer.param_groups[0]['lr']]
        else:
            result[epoch - 1, :] = [acc, mAUC, average_precision, average_recall, loss_meter.avg,
                                    valid_loss, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(eeg_model.state_dict(), "%s/models/best_eeg_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(eeg_model.state_dict(), "%s/models/eeg_model.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    ### save loss and acc loss
    import matplotlib.pyplot as plt

    # Extracting loss and valid_loss data from result
    loss_data = result[:, 4]  # Assuming loss is at index 5 in result
    valid_loss_data = result[:, 5]  # Assuming valid_loss is at index 6 in result

    plt.figure()
    # Plotting loss and valid_loss vs epochs
    plt.plot(range(1, len(loss_data) + 1), loss_data, label='Training Loss')
    plt.plot(range(1, len(valid_loss_data) + 1), valid_loss_data, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    if os.path.exists(exp_dir + '/result_plots') == False:
        os.mkdir(exp_dir + '/result_plots')
    plt.savefig(exp_dir + '/result_plots/loss.png')

    plt.figure()
    mAP_data = result[:, 0]  # Assuming mAP is at index 0 in result
    # Plotting mAP vs epochs
    plt.plot(range(1, len(mAP_data) + 1), mAP_data, label='mAP')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP vs Epochs')
    plt.legend()
    plt.savefig(exp_dir + '/result_plots/mAP.png')



def validate(eeg_model, val_loader, args, epoch, eval_target=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(eeg_model, nn.DataParallel):
        eeg_model = nn.DataParallel(eeg_model)
    eeg_model = eeg_model.to(device)
    # switch to evaluate mode
    eeg_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (eeg_input, labels) in enumerate(val_loader):
            eeg_input = eeg_input.to(device)
            # compute output
            eeg_output = eeg_model(eeg_input)
            predictions = eeg_output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)
            # compute the loss
            labels = labels.to(device)
            epsilon = 1e-7
            eeg_output = torch.clamp(eeg_output, epsilon, 1. - epsilon)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(eeg_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(eeg_output, labels)
            A_loss.append(loss.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        eeg_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = util.calculate_stats(eeg_output, target)
        # save the prediction here
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir + '/predictions') == False:
            os.mkdir(exp_dir + '/predictions')
            np.savetxt(exp_dir + '/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir + '/predictions/predictions_' + str(epoch) + '.csv', eeg_output, delimiter=',')
        # save the target for the separate eval set if there's one.
        if eval_target == True and os.path.exists(exp_dir + '/predictions/eval_target.csv') == False:
            np.savetxt(exp_dir + '/predictions/eval_target.csv', target, delimiter=',')
    return stats, loss





