import numpy as np
import pandas as pd
import random
import torch as torch
from torch import nn
import time
import configparser
from utils.log import trainingLogger
from utils.optimizer import Ranger
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchinfo import summary


def get_instance_from_section(section, *args):
    kwargs = {}
    for key, val in section:
        if key == 'func':
            func = eval(val)
        else:
            kwargs[key] = eval(val)
    return func(*args, **kwargs)


class ObjectiveFun:
    def __init__(self, fun, scaler=None, mask_flag=True):
        self.fun = fun
        self.scaler = scaler
        self.mask_flag = mask_flag

    def __call__(self, x, y):
        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)
            # y = self.scaler.inverse_transform(y)

        if self.mask_flag:
            mask = y > 1e-3
            # mask = y == y
            x = x[mask]
            y = y[mask]

        return self.fun(x, y)


def train_model(model, datasets, train_cfg_file, gpu, dataset_name):
    random.seed(0)
    cfg = configparser.ConfigParser()
    cfg.read('./config/training/{}.cfg'.format(train_cfg_file))

    print('==' * 25)
    print('training cfg file...')
    for section in cfg.sections():
        for key, val in cfg.items(section):
            print('{:20}\t{}'.format(key, val))
    print('==' * 25)

    train_set, valid_set, test_set = datasets

    enable_save_model = cfg.getboolean('other', 'save_model', fallback=False)
    model_name = model.name

    print("{} have {} parameters in total".format(model_name,
                                                  sum(x.numel() for x in model.parameters() if x.requires_grad)))
    gpu_list = [int(val) for val in gpu.split(',')]
    device = torch.device("cuda", gpu_list[0])
    model = nn.DataParallel(model.to(device), device_ids=gpu_list, output_device=gpu_list[0])

    optimizer = get_instance_from_section(cfg.items('optim'), model.parameters())
    scheduler = get_instance_from_section(cfg.items('scheduler'), optimizer)
    batch_size = cfg.getint('base', 'batch_size')
    epoch_num = cfg.getint('base', 'epoch_num')
    enable_noLoss = cfg.getboolean('base', 'enable_noLoss', fallback=False)
    enable_label_input = cfg.getboolean('base', 'enable_label_input', fallback=False)
    enable_reconstruct_x = cfg.getboolean('base', 'enable_reconstruct_x', fallback=False)
    enable_scheduled_sampling = cfg.getboolean('base', 'enable_scheduled_sampling', fallback=False)

    early_stop_patience = cfg.getint('base', 'early_stop_patience', fallback=999999)
    earlystop_metric_index = [int(val) for val in cfg.get('base', 'earlystop_metric_index', fallback='0').split(',')]
    visdom_ip = cfg.get('base', 'visdom_ip')

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_set, shuffle=False, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)

    scaler = train_set.scaler
    eval_metric_dict = {
        'rmse': ObjectiveFun(lambda x, y: np.sqrt(np.mean((x - y) ** 2)), mask_flag=False, scaler=scaler),
        'masked_rmse': ObjectiveFun(lambda x, y: np.sqrt(np.mean((x - y) ** 2)), mask_flag=True, scaler=scaler),
        'mae': ObjectiveFun(lambda x, y: np.mean(np.abs(x - y)), mask_flag=False, scaler=scaler),
        'masked_mae': ObjectiveFun(lambda x, y: np.mean(np.abs(x - y)), mask_flag=True, scaler=scaler),
        'mape': ObjectiveFun(lambda x, y: np.mean(np.abs(x - y) / y) * 100, mask_flag=False, scaler=scaler),
        'masked_mape': ObjectiveFun(lambda x, y: np.mean(np.abs(x - y) / y) * 100, mask_flag=True, scaler=scaler),
        'macro_f1': lambda x, y: -f1_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1),
                                           average='macro', zero_division=0),
        'micro_f1': lambda x, y: -f1_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1),
                                           average='micro', zero_division=0),
        'pre1': lambda x, y: -precision_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1),
                                              average=None, zero_division=0)[2],
        'pre2': lambda x, y: -precision_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1),
                                              average=None, zero_division=0)[1],
        'pre3': lambda x, y: -precision_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1),
                                              average=None, zero_division=0)[0],
        'rec1': lambda x, y: -
        recall_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0)[2],
        'rec2': lambda x, y: -
        recall_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0)[1],
        'rec3': lambda x, y: -
        recall_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0)[0],
        'macro_acc': lambda x, y: -accuracy_score(np.argmax(x, axis=-1).reshape(-1), np.argmax(y, axis=-1).reshape(-1)),
        'f': lambda x, y: -f1_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0),
        'f1': lambda x, y: -f1_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0)[
            2],
        'f2': lambda x, y: -f1_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0)[
            1],
        'f3': lambda x, y: -f1_score(np.argmax(x, axis=-1).reshape(-1), y.reshape(-1), average=None, zero_division=0)[
            0],
    }

    masked_flag = cfg.getboolean('obj', 'mask')
    loss_fn = eval(cfg.get('obj', 'training_loss'))
    enable_objective_fun_loss = cfg.getboolean('obj', 'enable_ObjectiveFun_loss', fallback=True)
    if enable_objective_fun_loss:
        loss_fn = ObjectiveFun(loss_fn, mask_flag=masked_flag, scaler=scaler)

    eval_metric_name_list = cfg.get('obj', 'eval_metric', fallback='')
    if len(eval_metric_name_list) != 0:
        eval_metric_name_list = eval_metric_name_list.split(',')
        eval_metric_list = [eval_metric_dict[name] for name in eval_metric_name_list]
    else:
        eval_metric_name_list = []
        eval_metric_list = []

    # 是否只保留验证集
    valid_only = cfg.getboolean('other', 'valid_only', fallback=False)
    feature_name = datasets[0].feature_name
    logger = trainingLogger(eval_metric_name_list, early_stop_patience, model_name, dataset_name, feature_name,
                            epoch_num=epoch_num, enable_save_model=enable_save_model, valid_only=valid_only)
    # for few shot
    enable_few_shot = cfg.getboolean('other', 'enable_few_shot', fallback=False)
    few_shot_ratio = cfg.getfloat('other', 'few_shot_ratio', fallback=0.2)
    stop_epoch = int(len(train_set) / batch_size * few_shot_ratio)

    for epoch in range(epoch_num):

        start_time = time.time()
        model.train()

        train_true_list = []
        train_preds_list = []

        valid_true_list = []
        valid_preds_list = []

        test_true_list = []
        test_preds_list = []

        for i, (x_batch, y_batch) in enumerate(train_loader):
            if i == 0 and epoch == 0:
                summary(model, x_batch.shape, verbose=1)
            if enable_few_shot:
                if i > stop_epoch:
                    break
            if enable_label_input:
                y_pred, y_batch = model(x_batch, y_batch)
                if enable_scheduled_sampling:
                    model.module.add_input_samples_num(1)
            else:
                y_pred = model(x_batch)

            if enable_reconstruct_x:
                y_batch = x_batch

            if enable_noLoss:
                if len(y_pred) == 2:
                    y_pred, loss = y_pred[0], y_pred[1]
                else:
                    y_pred, y_batch, loss = y_pred[0], y_pred[1], y_pred[2]
            try:
                loss = loss_fn(y_pred, y_batch.to(device))
            except:
                loss = loss_fn(y_pred.permute(0, 3, 1, 2), y_batch.squeeze(-1).long().to(device))
            optimizer.zero_grad()
            loss = torch.mean(loss)
            loss.backward()

            # for mi, m in enumerate(model.module.tower_list):
            #     tmp_val = torch.mean(torch.abs(m.ln.weight.grad)).cpu().numpy()
            #     grad_norm_list_dict[mi].append(tmp_val)

            # Clips gradient norm
            clip_grad_norm = cfg.getfloat('other', 'clip_grad_norm', fallback=None)
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            train_true_list.append(y_batch.detach().cpu())
            train_preds_list.append(y_pred.detach().cpu())

            if type(scheduler) == torch.optim.lr_scheduler.CyclicLR:
                scheduler.step()
                logger.update_lr(scheduler.get_last_lr())

        train_true = np.concatenate(train_true_list, axis=0)
        train_preds = np.concatenate(train_preds_list, axis=0)

        valid_start_time = time.time()

        model.eval()
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch)

            if enable_reconstruct_x:
                y_batch = x_batch

            if enable_noLoss:
                if len(y_pred) == 2:
                    y_pred, loss = y_pred[0], y_pred[1]
                else:
                    y_pred, y_batch, loss = y_pred[0], y_pred[1], y_pred[2]

            y_pred = y_pred.detach()

            valid_true_list.append(y_batch.detach().cpu())
            valid_preds_list.append(y_pred.detach().cpu())

        valid_true = np.concatenate(valid_true_list, axis=0)
        valid_preds = np.concatenate(valid_preds_list, axis=0)

        test_start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(test_loader):
            y_pred = model(x_batch)
            if enable_reconstruct_x:
                y_batch = x_batch

            if enable_noLoss:
                if len(y_pred) == 2:
                    y_pred, loss = y_pred[0], y_pred[1]
                else:
                    y_pred, y_batch, loss = y_pred[0], y_pred[1], y_pred[2]

            y_pred = y_pred.detach()

            test_true_list.append(y_batch.detach().cpu())
            test_preds_list.append(y_pred.detach().cpu())

        test_inference_time = time.time() - test_start_time

        test_true = np.concatenate(test_true_list, axis=0)
        test_preds = np.concatenate(test_preds_list, axis=0)

        valid_time = time.time() - valid_start_time
        elapsed_time = time.time() - start_time

        info_list = [elapsed_time, valid_time, test_inference_time]
        all_score = []

        for i, (preds, label) in enumerate([(train_preds, train_true), (valid_preds, valid_true),
                                            (test_preds, test_true)]):
            if valid_only and i != 1:
                continue
            score_list_list = []
            for metric in eval_metric_list:
                score_list = []
                if len(preds.shape) > 2:
                    for j in range(preds.shape[2]):
                        score = metric(preds[:, :, j], label[:, :, j])
                        score_list.append(score)
                else:
                    score = metric(preds, label)
                    score_list.append(score)

                score_list_list.append(score_list)
                mean_score = np.mean(score_list)
                info_list.append(mean_score)
            all_score.append(score_list_list)
        logger.addLine(info_list, model_state_dict=model.module.state_dict(), test_preds=test_preds)

        logger.addDetailedScores(all_score)

        if type(scheduler) != torch.optim.lr_scheduler.CyclicLR:
            if type(scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step()
            else:
                scheduler.step(logger.earlystop_scorelist[-1])

            logger.update_lr(scheduler.get_last_lr())

        if logger.early_stop():
            break

    logger.summary()
    return 'The process is over!'
