import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='DEKAN for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, default='DEKAN', help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DEKAN
parser.add_argument('--enc_in', type=int, default=7, help='number of input channels')
parser.add_argument('--e_layers', type=int, default=1, help='num of stacked encoder layers')
parser.add_argument('--n_branches', type=int, default=3, help='num of patch branches')
parser.add_argument('--patch_len_ls', type=str, default='16,48,96', help='patch length of each branch')
parser.add_argument('--stride_ls', type=str, default='8,24,48', help='stride of each branch')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--d_model', type=int, default=128, help='dimension of patch embedding')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--kernel_sizes', type=str, default='25,73,169', help='moving average kernels of the trend')
parser.add_argument('--period_list', type=str, default='24,168', help='seasonal periods')
parser.add_argument('--harmonics', type=int, default=2, help='Fourier harmonics per period')
parser.add_argument('--kan_basis', type=str, default='krawtchouk',
                    help='options:[krawtchouk, lucas, chebyshev, monomials, legendre, bernstein, hahn, bspline, fourier]')
parser.add_argument('--kan_degree', type=int, default=3, help='polynomial degree of the KAN layers')
parser.add_argument('--kan_grid_size', type=int, default=5, help='grid size of the bspline and fourier bases')

# ablations
parser.add_argument('--decomp_mode', type=str, default='full',
                    help='options:[full, none, residual_only, seasonal_only, trend_only]')
parser.add_argument('--kan_path', type=str, default='both', help='options:[both, ab, ba]')
parser.add_argument('--backbone_type', type=str, default='kan',
                    help='mixer inside each branch, options:[kan, transformer, smamba, mlpmixer, linear]')
parser.add_argument('--bottleneck_type', type=str, default='mlp', help='re-projection between stacked layers, options:[mlp, linear]')
parser.add_argument('--bottleneck_dim', type=str, default='0', help='hidden size of the mlp bottleneck, 0 for head_nf//2')
parser.add_argument('--loss', type=str, default='huber',
                    help='options:[huber, mse, mae, dbloss, fredf, transdf, psloss, tildeq, softdtw, dilate]')
parser.add_argument('--loss_base', type=str, default='mse', help='point-wise term of fredf, transdf and psloss')
parser.add_argument('--dbloss_beta', type=float, default=0.5, help='DBLoss seasonal weight')

# zero-shot
parser.add_argument('--target_data', type=str, default=None, help='evaluate the trained model on this dataset')
parser.add_argument('--target_data_path', type=str, default=None, help='data file of the target dataset')
parser.add_argument('--target_enc_in', type=int, default=None, help='number of channels of the target dataset')
parser.add_argument('--target_batch_size', type=int, default=None, help='test batch size on the target dataset')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--test_flop', action='store_true', default=False, help='params, FLOPs, peak memory and latency')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

# ablation flags that differ from their default are appended to the setting
ABLATIONS = [('decomp_mode', 'dc'), ('kan_basis', 'kb'), ('kan_grid_size', 'kg'), ('kan_degree', 'kd'),
             ('kan_path', 'kp'), ('backbone_type', 'bb'), ('bottleneck_type', 'bt'), ('bottleneck_dim', 'bn'),
             ('harmonics', 'h'), ('kernel_sizes', 'k'), ('period_list', 'p'), ('loss', 'ls'),
             ('random_seed', 'sd')]


def build_setting(ii):
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_df{}_el{}_br{}_pt{}_st{}_{}_{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.pred_len, args.d_model,
        args.d_ff, args.e_layers, args.n_branches, args.patch_len_ls.replace(' ', ''),
        args.stride_ls.replace(' ', ''), args.des, ii)
    for name, code in ABLATIONS:
        value = getattr(args, name)
        if value != parser.get_default(name):
            setting += '_{}{}'.format(code, str(value).replace(' ', ''))
    return setting


if args.is_training:
    for ii in range(args.itr):
        setting = build_setting(ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = build_setting(ii)

    exp = Exp(args)  # set experiments
    if args.test_flop:
        exp.test_flop()
    elif args.target_data:
        print('>>>>>>>zero-shot : {} -> {}<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting, args.target_data_path))
        exp.zero_shot(setting)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
