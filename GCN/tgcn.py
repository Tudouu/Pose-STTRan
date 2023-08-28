import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    def __init__(self,
                 in_channels,#3
                 out_channels,#64
                 kernel_size,#3
                 h_kernel_size=1,
                 w_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,#3
            out_channels * kernel_size,#192
            kernel_size=(h_kernel_size,w_kernel_size),#后边要变成2
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        print(x.size())
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)#(1,3,64,num,18)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A