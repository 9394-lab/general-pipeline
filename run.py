import argparse
import configparser
import torch
from hyperopt import fmin, tpe, hp
from dataset.process import read_dataset
from model import build_model, train_model

if __name__ == '__main__':
    # 读取命令
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataset", type=str, default='dataset')
    parser.add_argument("--model", type=str, default='MLP')
    parser.add_argument("--train", type=str, default='TimesNet')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--tpe", type=bool, default=False)
    args = parser.parse_args()
    print(args)

    dataset = read_dataset(args.dataset)

    if args.tpe:
        def objective(tpe_args):
            try:
                model = build_model(dataset, args.model, tpe_args)
                score = train_model(model, dataset, args.train, args.gpu, args.dataset)
            except BaseException as e:
                print(e)
                return 999
            torch.cuda.empty_cache()
            return score


        space = {
            # 'patch_len': hp.choice('patch_len', (6)),

            'seg_len': hp.choice('seg_len', (1, 2, 3, 4)),
            # 'out_channels': hp.choice('out_channels', (16, 32, 48, 64)),
            'stride': hp.choice('stride', (1, 2, 3)),
            # 'factor': hp.choice('factor', (4, 8, 16)),
            # 'win_size': hp.choice('win_size', (1, 2, 3, 4)),
            'num_blocks': hp.choice('num_blocks', (1, 2, 3, 4)),
            'out_channels': hp.choice('out_channels', (16, 32, 48, 64, 96)),
            'dropout': hp.choice('dropout', (0.2, 0.3)),
            # 'weight': hp.choice('weight', (0.2, 0.5, 1, 1.5, 2))
        }
        max_eval = 30

        best = fmin(objective, space, max_evals=max_eval, algo=tpe.suggest)
        print('Best hyper-parameters are:')
        print(best)

    else:
        model = build_model(dataset, args.model)
        model = train_model(model, dataset, args.train, args.gpu, args.dataset)
