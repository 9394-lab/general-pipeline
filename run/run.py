import argparse
import configparser
from hyperopt import fmin, tpe, hp
import torch
from dataset.process import read_dataset
from model import build_model, train_model

if __name__ == '__main__':
    # 读取命令
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dataset", type=str, default='dataset')
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--tpe", type=bool, default=False)
    args = parser.parse_args()
    print(args)

    dataset = read_dataset(args.dataset)

    if args.tpe:
        def objective(tpe_args):
            try:
                model = build_model(dataset, args.model, tpe_args)
                score = train_model(model, dataset, args.train, args.gpu)
            except BaseException as e:
                print(e)
                return 999
            torch.cuda.empty_cache()
            return score

        space = {
            # 'dqk': hp.choice('dqk', (4, 6, 8)),
            # 'node_embedding_dim': hp.choice('node_embedding_dim', (4, 8, 12, 16)),
            'num_blocks': hp.choice('num_blocks', (1, 2, 4)),
            'out_channels': hp.choice('out_channels', (32, 16, 48, 64)),
            # 'heads': hp.choice('heads', (4, 8)),
            'dropout': hp.choice('dropout', (0.0, 0.1, 0.2)),
            # 'skip_channels': hp.choice('skip_channels', (64, 128, 256)),
            # 'residual_channels': hp.choice('residual_channels', (32, 64, 128)),
            # 'num_classes': hp.choice('num_classes', (4, 8))
        }
        max_eval = 10

        best = fmin(objective, space, max_evals=max_eval, algo=tpe.suggest)
        print('Best hyper-parameters are:')
        print(best)

    else:
        model = build_model(dataset, args.model)
        model = train_model(model, dataset, args.train, args.gpu)
