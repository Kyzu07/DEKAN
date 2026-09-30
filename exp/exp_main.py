from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DEKAN
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings

warnings.filterwarnings('ignore')


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = DEKAN.Model(self.args).float()
        print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters())))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, train_data=None):
        loss = self.args.loss
        if loss == 'huber':
            return nn.HuberLoss(delta=0.5)
        if loss == 'mse':
            return nn.MSELoss()
        if loss == 'mae':
            return nn.L1Loss()
        if loss == 'dbloss':
            from utils.losses import DBLoss
            return DBLoss(beta=self.args.dbloss_beta)
        if loss == 'fredf':
            from utils.losses import FreDFLoss
            return FreDFLoss(base=self.args.loss_base)
        if loss == 'transdf':
            from utils.losses import TransDFLoss
            from utils.pca_basis import BasisCache, fit_or_load_pca_base
            return TransDFLoss(BasisCache(*fit_or_load_pca_base(train_data, self.args)), base=self.args.loss_base)
        if loss == 'psloss':
            from utils.losses import PSLoss
            return PSLoss(self.model, base=self.args.loss_base)
        if loss == 'a1':
            from utils.losses import A1Loss
            return A1Loss(self.model, w_aux=self.args.a1_w_aux)
        if loss == 'tildeq':
            from utils.losses_shape import TILDEQLoss
            return TILDEQLoss()
        if loss == 'softdtw':
            from utils.losses_shape import SoftDTWLoss
            return SoftDTWLoss()
        if loss == 'dilate':
            from utils.losses_shape import DILATELoss
            return DILATELoss()
        raise ValueError('Unknown loss: {}'.format(loss))

    def _load_best_ckpt(self, path):
        ckpt = os.path.join(path, 'checkpoint_ema.pth')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(ckpt))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs, _ = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
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

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train_data)
        vali_criterion = nn.MSELoss()

        # validation and checkpoints use EMA weights
        ema = EMA(self.model, decay=0.999)
        best_val = float('inf')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, aux = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                else:
                    outputs, aux = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    if self.args.loss == 'a1':
                        loss = criterion(outputs, batch_y, aux)
                    else:
                        loss = criterion(outputs, batch_y)
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
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            ema.apply_to(self.model)
            vali_loss = self.vali(vali_data, vali_loader, vali_criterion)
            test_loss = self.vali(test_data, test_loader, vali_criterion)
            ema.restore(self.model)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if vali_loss < best_val:
                best_val = vali_loss
                ema.apply_to(self.model)
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint_ema.pth'))
                ema.restore(self.model)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        self._load_best_ckpt(path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self._load_best_ckpt(os.path.join(self.args.checkpoints, setting))

        preds = []
        trues = []
        inputx = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs, _ = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                inputx.append(batch_x.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

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

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        return

    def zero_shot(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        ckpt = os.path.join(path, 'checkpoint_ema.pth')
        if not os.path.exists(ckpt):
            ckpt = os.path.join(path, 'checkpoint.pth')
        state = torch.load(ckpt)

        # rebuild for the target channels; the AR trend head is the only per-channel weight
        self.args.data = self.args.target_data
        self.args.data_path = self.args.target_data_path
        self.args.enc_in = self.args.target_enc_in
        self.args.batch_size = self.args.target_batch_size or self.args.batch_size
        self.model = self._build_model().to(self.device)
        key = 'model.ar_head.proj.weight'
        if key in state and state[key].shape[0] != self.args.enc_in:
            state[key] = state[key].mean(dim=0, keepdim=True).expand(self.args.enc_in, -1, -1).clone()
        self.model.load_state_dict(state)

        target = os.path.splitext(self.args.target_data_path)[0]
        self.test(setting + '_to_' + target)

    def test_flop(self):
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in, device=self.device)
        params = sum(p.numel() for p in self.model.parameters())
        self.model.eval()
        with torch.no_grad():
            counter = FlopCounterMode(display=False)
            with counter:
                self.model(x)
            flops = counter.get_total_flops()
            for _ in range(10):
                self.model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.model(x)
            torch.cuda.synchronize()
            memory = torch.cuda.max_memory_allocated() / 1e9
            start = time.time()
            for _ in range(50):
                self.model(x)
            torch.cuda.synchronize()
            latency = (time.time() - start) / 50 * 1000
        print('params: {:.3f}M, FLOPs: {:.3f}G, peak memory: {:.3f}GB, latency: {:.2f}ms/batch'.format(
            params / 1e6, flops / 1e9, memory, latency))

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            self._load_best_ckpt(os.path.join(self.args.checkpoints, setting))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs, _ = self.model(batch_x)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        return
