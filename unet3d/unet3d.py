import torch
import torch.nn as nn
NORM_CLS = nn.InstanceNorm3d
ACT_MODULE = nn.ReLU(inplace=True)


class UNet3d(nn.Module):
    def __init__(self, c_in=1, c_out=2, c_start=8, layers=4, kernel=3, pad_in=None, pad_out=0,
                 p_dropout=.0, norm_cls=NORM_CLS, act_module=ACT_MODULE, cs=None):
        super(UNet3d, self).__init__()
        cs = [(c_start * 2 ** i, c_start * 2 ** (i + 1)) for i in range(layers)] if cs is None else cs
        self.inp = DoubleConv(c_in, c_start, kernel, p_dropout=p_dropout, norm_cls=norm_cls, act_module=act_module, pad=pad_in)
        self.downs = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(cs):
            self.downs.append(Down(ch_in, ch_out, kernel, p_dropout=p_dropout, norm_cls=norm_cls, act_module=act_module))
        self.ups = nn.ModuleList()
        for i, (ch_in, ch_out) in enumerate(cs[::-1]):
            self.ups.append(Up(ch_out, ch_in, kernel, p_dropout=p_dropout, norm_cls=norm_cls, act_module=act_module))
        self.out = nn.Conv3d(c_start, c_out, 1, padding=pad_out)

    def forward(self, x):
        xs = []
        x = self.inp(x)
        for down in self.downs:
            xs.append(x)
            x = down(x)
        for up in self.ups:
            x = up(x, xs.pop())
        return self.out(x)


class Down(nn.Sequential):
    def __init__(self, c_in, c_out, kernel=3, p_dropout=.0, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        super(Down, self).__init__(DoubleConv(c_in, c_out, kernel, p_dropout, norm_cls, act_module), nn.MaxPool3d(2))


class Up(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, p_dropout=.0, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        super(Up, self).__init__()
        self.up_conv = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.conv = DoubleConv(c_in, c_out, kernel, p_dropout=p_dropout, norm_cls=norm_cls, act_module=act_module)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, p_dropout=.0, norm_cls=NORM_CLS, act_module=ACT_MODULE, pad=None):
        pad1 = pad if pad is not None else kernel // 2
        conv1 = [nn.Conv3d(in_ch, out_ch, kernel, padding=pad1), norm_cls(num_features=out_ch), act_module]
        if p_dropout > 0:
            conv1.append(nn.Dropout3d(p_dropout))
        conv2 = [nn.Conv3d(out_ch, out_ch, kernel, padding=kernel // 2), norm_cls(num_features=out_ch), act_module]
        super(DoubleConv, self).__init__(*conv1, *conv2)
  
