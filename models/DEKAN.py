__all__ = ['DEKAN']

from torch import nn

from layers.DEKAN_backbone import DEKAN_backbone


def _ints(s):
    return [int(i) for i in str(s).split(',')]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        patch_len = _ints(configs.patch_len_ls)
        stride = _ints(configs.stride_ls)

        self.model = DEKAN_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                    patch_len=patch_len, stride=stride, n_layers=configs.e_layers,
                                    n_branches=configs.n_branches, d_model=configs.d_model, d_ff=configs.d_ff,
                                    dropout=configs.dropout, head_dropout=configs.head_dropout,
                                    padding_patch=configs.padding_patch, revin=configs.revin,
                                    affine=configs.affine, subtract_last=configs.subtract_last,
                                    decomp_mode=configs.decomp_mode, kernel_sizes=_ints(configs.kernel_sizes),
                                    periods=_ints(configs.period_list), harmonics=configs.harmonics,
                                    backbone_type=configs.backbone_type, kan_basis=configs.kan_basis,
                                    kan_degree=configs.kan_degree, kan_grid_size=configs.kan_grid_size,
                                    kan_path=configs.kan_path, bottleneck_type=configs.bottleneck_type,
                                    bottleneck_dim=configs.bottleneck_dim)

    def forward(self, x):                                                   # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)                                              # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0, 2, 1)                                              # x: [Batch, Output length, Channel]
        return x
