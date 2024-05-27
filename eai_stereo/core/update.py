import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from mogrifier import Mogrifier


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):  # h 上一层的输入 z update r reset
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)  # q = h'

        h = (1 - z) * h + z * q
        return h


class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(LSTM, self).__init__()
        self.conv_it = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_c_t = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_ft = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_ot = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.m = Mogrifier(
            dim=hidden_dim,
            iters=5,  # number of iterations, defaults to 5 as paper recommended for LSTM
            factorize_k=None,  # factorize weight matrices into (dim x k) and (k x dim), if specified
            kernel_size=3
        )
        # print('hidden_dim')
        # print(hidden_dim)
        # print('input_dim')
        # print(input_dim)

    def forward(self, c, h, bi, bf, bc, bo, *x_list):  # h 上一层的输入 z update r reset cr->bf cz->bi cq->bc bo
        # print('len:x_list')
        # print(len(x_list))
        # print('x.shape')
        # print(x.shape)
        # print('h.shape')
        # print(h.shape)
        x = torch.cat(x_list, dim=1)  # 这里把lstm(net[],net,*inp以后所有的东西都拼起来 因为GRU只有h LSTM有c h所以后面多了一个(128)
        # exit()
        if x.shape[1] == h.shape[1]:
            x, h = self.m(x, h)
            # x = x + new_x
            # h = h + new_h
        hx = torch.cat([h, x], dim=1)
        # print('hx.shape')
        # print(hx.shape)
        # TODO 
        ft = torch.sigmoid(self.conv_ft(hx) + bf)  # bf是bias ft = forget_gate
        it = torch.sigmoid(self.conv_it(hx) + bi)
        c_t = torch.tanh(self.conv_c_t(hx) + bc)
        ct = c * ft + it * c_t
        ot = torch.sigmoid(self.conv_ot(hx) + bo)
        ht = ot * torch.tanh(ct)
        return ct, ht


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


def interpTuple(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x[0], dest.shape[2:], **interp_args), F.interpolate(x[1], dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2 ** self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow


class LSTMMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.lstm08 = LSTM(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.lstm16 = LSTM(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.lstm32 = LSTM(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2 ** self.args.n_downsample

        # print('(factor ** 2) * 9')
        # print((factor ** 2) * 9)

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))

    def forward(self, netC, netH, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        # print('inp[2]')
        # print(inp[2])
        # print('len(inp[2])')  # inp[?] = 3
        # print(len(inp[2]))
        # print('```net[0]')
        # print(net[0].shape)
        # print('```net[1]')
        # print(net[1].shape)
        if iter32:
            netC[2], netH[2] = self.lstm32(netC[2], netH[2], *(inp[2]), pool2x(netH[1]))
            # print(type(net[2]))
            # print(len(net[2]))
            # print('net[2]```')
            # print(net[2].shape)
            # print('net[1]```')
            # print(net[1].shape)

            # a, b = self.lstm32(net[2], net[2], *(inp[2]), pool2x(net[1]))
            # print('a```')
            # print(a.shape)
            # print('b```')
            # print(b.shape)
        if iter16:
            if self.args.n_gru_layers > 2:
                netC[1], netH[1] = self.lstm16(netC[1], netH[1], *(inp[1]), pool2x(netH[0]),
                                               interp(netH[2], netH[1]))
            else:
                netC[1], netH[1] = self.lstm16(netC[1], netH[1], *(inp[1]), pool2x(netH[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                netC[0], netH[0] = self.lstm08(netC[0], netH[0], *(inp[0]), motion_features,
                                               interp(netH[1], netH[0]))
            else:
                netC[0], netH[0] = self.lstm08(netC[0], netH[0], *(inp[0]), motion_features)

        if not update:
            return netH
        delta_flow = self.flow_head(netH[0])
        # scale mask to balence gradients
        # print('###net[0].shape')
        # print(netH[0].shape)
        mask = .25 * self.mask(netH[0])
        # print('Success!!!!!!!!!!!!!!')
        return netC, netH, mask, delta_flow
