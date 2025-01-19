import torch.nn as nn
NORM_CLS = nn.InstanceNorm3d
ACT_MODULE = nn.ReLU(inplace=True)


class LinkNet3d(nn.Module):
    def __init__(self, c_in=1, c_out=2, c_start=16, layers=4, kernel=3, init_kernel=7,
                 bias=False, norm_cls=NORM_CLS, act_module=ACT_MODULE, cs=None):
        super(LinkNet3d, self).__init__()
        if cs is None:
            cs = [(c_start, c_start)] + [(c_start * 2**i, c_start * 2**(i+1)) for i in range(layers - 1)]
        self.init = nn.Sequential(nn.Conv3d(c_in, cs[0][0], init_kernel, stride=2, padding=init_kernel//2, bias=bias),
                                  norm_cls(cs[0][0]), act_module, nn.MaxPool3d(3, stride=2, padding=1))
        strides = [1] + [2] * (layers - 1)
        self.downs = nn.ModuleList()
        for i, ((ch_in, ch_out), stride) in enumerate(zip(cs, strides)):
            self.downs.append(Down(ch_in, ch_out, kernel, stride=stride, pad=kernel//2,
                                   norm_cls=norm_cls, act_module=act_module))
        output_paddings = [0] + [1] * (layers - 1)
        self.ups = nn.ModuleList()
        for i, ((ch_in, ch_out), stride, pad_out) in enumerate(zip(cs[::-1], strides[::-1], output_paddings[::-1])):
            self.ups.append(Up(ch_out, ch_in, kernel, stride=stride, pad=kernel//2,
                               pad_out=pad_out, norm_cls=norm_cls, act_module=act_module))
        tp_conv1 = nn.Sequential(nn.ConvTranspose3d(c_start, c_start//2, kernel, stride=2, padding=1, output_padding=1),
                                 norm_cls(num_features=c_start//2), act_module)
        conv = nn.Sequential(nn.Conv3d(c_start//2, c_start//2, 3, stride=1, padding=1),
                             norm_cls(num_features=c_start//2), act_module)
        tp_conv2 = nn.ConvTranspose3d(c_start//2, c_out, 2, stride=2, padding=0)
        self.out = nn.Sequential(tp_conv1, conv, tp_conv2)

    def forward(self, x):
        xs = []
        x = self.init(x)
        for down in self.downs:
            xs.append(x)
            x = down(x)
        for up in self.ups:
            x = up(x) + xs.pop()
        return self.out(x)


class Down(nn.Sequential):
    def __init__(self, c_in, c_out, kernel, stride=1, pad=0, bias=False, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        block1 = Block(c_in, c_out, kernel, stride=stride, pad=pad, bias=bias, norm_cls=norm_cls, act_module=act_module)
        block2 = Block(c_out, c_out, kernel, stride=1, pad=pad, bias=bias)
        super(Down, self).__init__(block1, block2)


class Up(nn.Sequential):
    def __init__(self, c_in, c_out, kernel, stride=1, pad=0, pad_out=0,
                 bias=False, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        conv1 = nn.Sequential(nn.Conv3d(c_in, c_in//4, 1, bias=bias), norm_cls(c_in // 4), act_module)
        tp_conv = nn.Sequential(nn.ConvTranspose3d(c_in//4, c_in//4, kernel, stride=stride, padding=pad, output_padding=pad_out, bias=bias),
                                norm_cls(c_in//4), act_module)
        conv2 = nn.Sequential(nn.Conv3d(c_in//4, c_out, 1, stride=1, padding=0, bias=bias),
                              norm_cls(c_out), act_module)
        super(Up, self).__init__(conv1, tp_conv, conv2)


class Block(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride=1, pad=0, bias=False, norm_cls=NORM_CLS, act_module=ACT_MODULE):
        super(Block, self).__init__()
        conv1 = [nn.Conv3d(c_in, c_out, kernel, stride=stride, padding=pad, bias=bias), norm_cls(c_out), act_module]
        conv2 = [nn.Conv3d(c_out, c_out, kernel, stride=1, padding=pad, bias=bias), norm_cls(c_out)]
        self.double_conv = nn.Sequential(*conv1, *conv2)
        if stride > 1:
            self.down = nn.Sequential(nn.Conv3d(c_in, c_out, 1, stride=stride, bias=False), norm_cls(c_out))
        else:
            self.down = None
        self.act = act_module

    def forward(self, x):
        residual = x if self.down is None else self.down(x)
        x = self.double_conv(x)
        x += residual
        return self.act(x)
  
