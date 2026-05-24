from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, MTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
# import timm
# from timm.scheduler import CosineLRScheduler, PlateauLRScheduler

import warnings
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import pickle as pkl

from einops import rearrange

warnings.filterwarnings('ignore')


# === EMA helper ===
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # shadow stores a copy of each trainable parameter
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply_to(self, model):
        # swap model params -> EMA weights (keep originals in backup)
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        # restore original (non-EMA) weights
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

# ---- Huber loss (a.k.a. Smooth L1) ----
def huber_loss(pred, target, seq_len):
    delta = 0.5
    # if seq_len == 96:
    #     delta = 0.5
    # elif seq_len == 192:
    #     delta = 0.8
    # elif seq_len == 336:
    #     delta = 1.2
    # else:
    #     delta = 1.5

    e = pred - target
    abs_e = e.abs()
    quad  = torch.minimum(abs_e, torch.tensor(delta, device=e.device))
    lin   = abs_e - quad
    return (0.5 * quad**2 + delta * lin).mean()


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        cfg = vars(args)
        self.use_mlflow = self.args.use_mlflow
        if self.use_mlflow:
            name = self.args.model_id
            project = self.args.mlflow_project
            experiment = mlflow.set_experiment(project)
            mlflow.start_run(run_name=name)
            mlflow.log_params(cfg)



    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'MTST':MTST
        }
        
        # ...


        model = model_dict[self.args.model].Model(self.args).float().cuda()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _load_best_ckpt(self, path):
        ema_path = os.path.join(path, 'checkpoint_ema.pth')
        std_path = os.path.join(path, 'checkpoint.pth')
        ckpt_path = ema_path if os.path.exists(ema_path) else std_path
        self.model.load_state_dict(torch.load(ckpt_path))
        return ckpt_path

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        # model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                        if isinstance(outputs, (tuple, list)):
                            outputs, draw_list, attn_list = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                if 'TST' in self.args.model:
                    outputs = outputs[:, :, f_dim:]
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # # to continue mtst on traffic
        # print('loading model')
        # model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(model_path))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        ema = EMA(self.model, decay=0.999)   # or expose decay via args
        best_val = float('inf')              # for saving an EMA checkpoint on improvement


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        if 'timm' in self.args.lradj:
            if 'cos' in self.args.lradj:
                scheduler = CosineLRScheduler(optimizer = model_optim,
                                              t_initial=train_steps - self.args.warmup_steps,
                                              lr_min=1e-8,
                                              warmup_t=self.args.warmup_steps,
                                              warmup_prefix=True,
                                              warmup_lr_init=1e-8
                                              )
            elif 'plateau' in self.args.lradj:
                scheduler = PlateauLRScheduler(optimizer=model_optim,
                                               decay_rate=self.args.decay_rate,
                                               patience_t=self.args.lr_patience,
                                               warmup_t=self.args.warmup_steps,
                                               warmup_lr_init=1e-8,
                                               lr_min=self.args.lr_min,
                                               mode='min'
                                        )

        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)

        start_temp = 1.0
        min_temp = 0.001
        num_steps = self.args.train_epochs * len(train_loader)
        current_step = 0
        temperature = start_temp

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                # Calculate the current temperature based on the step
                annealing_factor = max(min_temp, 1 - (current_step / num_steps))
                temperature = start_temp * annealing_factor

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # loss = criterion(outputs, batch_y)
                        loss = huber_loss(outputs, batch_y, self.args.seq_len)

                        train_loss.append(loss.item())
                else:
                    # print('ho mama')
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            # print('ki kos') eidai. eikhanei hoy kaam
                            outputs = self.model(batch_x, temperature)

                            if isinstance(outputs, (tuple, list)):
                                outputs, draw_list, attn_list = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # loss = criterion(outputs, batch_y)
                    loss = huber_loss(outputs, batch_y, self.args.seq_len)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                ema.update(self.model)
                current_step += 1    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)

            # evaluate using EMA weights
            ema.apply_to(self.model)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            ema.restore(self.model)


            if self.use_mlflow:
                log_dict = {'train/loss': train_loss, 'vali/loss': vali_loss, 'test/loss': test_loss}
                mlflow.log_metrics(log_dict, step=epoch)
                lr = model_optim.param_groups[0]['lr']
                mlflow.log_metric('lr', lr, step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # save EMA weights when vali improves
            if vali_loss < best_val:
                best_val = vali_loss
                ema.apply_to(self.model)
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint_ema.pth'))
                ema.restore(self.model)


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if ('timm' in self.args.lradj and 'plateau' in self.args.lradj \
                    and model_optim.param_groups[0]['lr'] > self.args.lr_min) \
                    or epoch + 1 < self.args.warmup_steps:
                early_stopping.counter=0
                print("Reset Early stopping before Plateau Scheduler reach the lr_min")

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, metric=vali_loss)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))
        # Prefer EMA weights for the final in-memory model
        self._load_best_ckpt(path)
        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            # ckpt_dir = os.path.join('./checkpoints', setting)
            # ema_path = os.path.join(ckpt_dir, 'checkpoint_ema.pth')
            # std_path = os.path.join(ckpt_dir, 'checkpoint.pth')
            # path_to_load = ema_path if os.path.exists(ema_path) else std_path
            # self.model.load_state_dict(torch.load(path_to_load))

            ckpt_dir = os.path.join('./checkpoints', setting)
            self._load_best_ckpt(ckpt_dir)


        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        # visual always true in test function:
        self.args.visual = True

        if self.args.visual:
            self.model.model.visual = True

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                            draw_list = None
                            attn_list = None
                            if isinstance(outputs, (tuple, list)):
                                outputs, draw_list, attn_list = outputs
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())


                ### draw figures ########

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if self.args.features == "S":
                #         variate_ids = [0]
                #     else:
                #         # variate_ids = [11,12,13,14]
                #         variate_ids = [0,1,2,6]
                    # for j in variate_ids:
                    #     gt = np.concatenate((input[0, :, j], true[0, :, j]), axis=0)
                    #     pd = pred[0, :, j]
                    #     # pd = np.concatenate((input[0, :, j], pred[0, :, j]), axis=0)
                    #     # time_step = np.arange(input.shape[1], gt.shape[1])
                    #     visual(gt, pd, os.path.join(folder_path, str(i) + f'_var{j}.pdf'))




        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = rearrange(preds, 'b l d -> b d l')
        trues = rearrange(trues, 'b l d -> b d l')
        inputx = rearrange(inputx, 'b l d -> b d l')


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.use_mlflow:
            log_dict= {'best/test_mae': mae, 'best/test_mse': mse, "best/test_rse": rse}
            mlflow.log_metrics(log_dict)
            mlflow.end_run()


        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self._load_best_ckpt(path)
            # path = os.path.join(self.args.checkpoints, setting)
            # best_model_path = path + '/' + 'checkpoint.pth'
            # self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        # self.model.model.visual = True

        return
