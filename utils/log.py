# from matplotlib import pyplot as plt
import numpy as np
# import visdom
import time
import torch


def timelimit(limit=1):
    def timelimit_wrapper(func):
        def wrapped_func(*arg, **kwargs):
            now = time.time()
            elasped_time = now - wrapped_func.last_time
            if elasped_time > limit:
                wrapped_func.last_time = now
                return func(*arg, **kwargs)
            else:
                return

        setattr(wrapped_func, 'last_time', 0)
        return wrapped_func

    return timelimit_wrapper


class trainingLogger:
    def __init__(self, metric_names, earlystop_rounds, model_name, dataset_name, feature_name, epoch_num=9999,
                 enable_save_model=False, valid_only=False,
                 visdom_ip='127.0.0.1',
                 visdom_port='20',
                 visdom_env='main'):
        self.metric_names = metric_names
        self.earlystop_rounds = earlystop_rounds
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.feature_name = feature_name
        self.enable_save_model = enable_save_model
        self.valid_only = valid_only
        self.all_infolist = []
        self.baseLogFormat = 'Epoch {:>5d}/' + str(epoch_num) + \
                             '\t| total_time_cost: {:5.2f}\t| valid_loop_cost: {:5.2f}\t| test_loop_cost: {:5.2f}\t| '
        self.log_list = []

        self.count = 0

        if valid_only:
            earlystop_metric_index = [0]
            self.metrics_list = ['valid']
        else:
            earlystop_metric_index = [1]
            self.metrics_list = ['train', 'valid', 'test']

        # base + 3 * index: 只有valid则index为0 否则为1 要把train的跳过
        self.es_read_index = [3 + 3 * val for val in earlystop_metric_index]
        self.best_score = [99999] * len(earlystop_metric_index)
        self.es_count = [0] * len(earlystop_metric_index)
        self.best_index = -1
        self.earlystop_scorelist = [[] for i in range(len(earlystop_metric_index))]
        self.best_model_state_dict = None

        # self.vis = visdom.Visdom(server=visdom_ip, port=visdom_port, env=visdom_env)
        self.lr_list = []

        self.detailed_scores = []

    def addLine(self, infolist, model_state_dict=None, test_preds=None):
        self.count += 1
        baseLog = self.baseLogFormat.format(self.count, *infolist[0:3])
        scorelist = infolist[3:]

        for i, metric_str in enumerate(self.metric_names):
            baseLog += metric_str + '\t'
            for j, loop_name in enumerate(self.metrics_list):
                baseLog += '{}:\t{:>9.3f}\t '.format(loop_name, scorelist[i + j*3])

            # y_in = np.array([scorelist[(i * 3):(i * 3 + 3)]])
            # x_in = np.array([self.count])
            # self.__metric_plot(y_in, x_in, metric_str)

            baseLog += '| '
        print(baseLog)
        self.log_list.append(baseLog)
        self.all_infolist.append(infolist)

        for i in range(len(self.es_read_index)):
            earlyst_score = infolist[self.es_read_index[i]]
            self.earlystop_scorelist[i].append(earlyst_score)

            if earlyst_score < self.best_score[i]:
                self.es_count[i] = 0
                self.best_score[i] = earlyst_score
                self.best_index = self.count - 1
                self.best_model_state_dict = model_state_dict
                if self.count > 5 and self.enable_save_model:
                    if self.valid_only:
                        path = './save/{}_epoch_{}_f1_{:.3f}_on_PeMS2019'.format(self.model_name, self.count,
                                                                                 -infolist[3])
                    else:
                        path = './save/{}_epoch_{}_MAE_{:.3f}_on_{}_{}'.format(self.model_name, self.count,
                                                                               infolist[-3],
                                                                               self.dataset_name, self.feature_name)
                    torch.save(self.best_model_state_dict, path+'.pt')
                    np.save(path+'.npy', test_preds)
            else:
                self.es_count[i] += 1
                print('not update for {} epochs'.format(self.es_count[i]))

    def early_stop(self):
        for i in range(len(self.es_read_index)):
            assert self.es_count[i] <= self.earlystop_rounds, 'Exists Unnecessary training!!!{}/{}'.format(
                self.es_count,
                self.earlystop_rounds)
            if self.es_count[i] == self.earlystop_rounds:
                return True
        return False

    @timelimit(1)
    def __lr_plot(self, y_in, x_in):
        update = 'append' if len(self.lr_list) > 1 else 'replace'
        self.vis.line(y_in, x_in, update=update, win='trainning_lr',
                      opts={'title': 'learning_rate',
                            'ylabel': 'lr'})

    # @timelimit(1)
    def __metric_plot(self, y_in, x_in, name):
        update = 'append' if self.count > 1 else 'replace'
        self.vis.line(y_in, x_in, update=update, win=name,
                      opts={'title': name,
                            'ylabel': name,
                            'legend': ['train', 'valid', 'test']})

    def update_lr(self, lr):
        x = np.array([len(self.lr_list)])
        y = np.array(lr)
        self.lr_list.append(lr)
        # self.__lr_plot(y, x)

    def addDetailedScores(self, scores):
        self.detailed_scores.append(scores)

    def plot(self):
        pass

    def summary(self):
        print('==' * 20)
        print('best epoch:\t')
        print(self.log_list[self.best_index])
        print('--' * 20)
        best_detailed_scores = self.detailed_scores[self.best_index]
        best_detailed_scores = np.array(best_detailed_scores)  # metrics * datasets * horizon

        cell_size = '15'
        header_str = ('{:>' + cell_size + '}\t|').format('horizon')
        for j in range(best_detailed_scores.shape[0]):
            for k in range(best_detailed_scores.shape[1]):
                header_str += ('{:>' + cell_size + '}\t').format(self.metrics_list[j])
            header_str += '|'
        print(header_str)

        header_str = ('{:>' + cell_size + '}\t|').format('horizon')
        for j in range(best_detailed_scores.shape[0]):
            for k in range(best_detailed_scores.shape[1]):
                header_str += ('{:>' + cell_size + '}\t').format(self.metric_names[k])
            header_str += '|'
        print(header_str)

        for i in range(best_detailed_scores.shape[-1]):
            str_out = ('{:>' + cell_size + '}\t|').format(i)
            for j in range(best_detailed_scores.shape[0]):
                for k in range(best_detailed_scores.shape[1]):
                    str_out += ('{:>' + cell_size + '.3f}\t').format(abs(best_detailed_scores[j, k, i]))
                str_out += '|'
            print(str_out)

        print('==' * 20)
        # return best_detailed_scores[0][2].mean()
