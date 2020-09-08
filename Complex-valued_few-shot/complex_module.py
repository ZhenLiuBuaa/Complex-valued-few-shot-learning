import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter


def conv_weight(in_planes, planes, kernel_size=3, stride=1, padding=0, bias=False, transpose=False):
    " init convolutions parameters, necessary due to code architecture "
    if transpose:
        params = nn.ConvTranspose2d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=bias).weight.data
    else:
        params = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=bias).weight.data
    return params


class Complex(nn.Module):
    def __init__(self, real=None, imag=None):
        super(Complex, self).__init__()
        self.real = real
        if imag is None and real is not None:
            self.imag = torch.zeros_like(self.real)
        elif imag is None and real is None:
            self.imag = None
        else:
            self.imag = imag
        if self.real is not None:
            self.shape = self.real.shape
            self.size = self.real.size
        else:
            self.shape = None
            self.size = None

    def mag(self):
        return torch.sqrt(self.real ** 2 + self.imag ** 2 + np.finfo(float).eps)

    def phase(self):
        return torch.atan2(self.imag, self.real)

    def from_polar(self, mag, phase):
        self.real = mag * torch.cos(phase)
        self.imag = mag * torch.sin(phase)
        return

    def view(self, *params):
        return Complex(self.real.view(*params), self.imag.view(*params))

    def __repr__(self):
        # print(f'Complex Variable containing:\nreal:\n{self.real}imaginary:\n{self.imag}') <- does'nt work
        return ''

class C_MaxPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                ceil_mode=False):
        super(C_MaxPooling, self).__init__()
        self.Max = nn.MaxPool2d(kernel_size, stride=stride, padding=padding,
                                ceil_mode=ceil_mode)
    def forward(self, x):
        return Complex(self.Max(x.real),
                       self.Max(x.imag)
                       )
class C_AvePooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                ceil_mode=False):
        super(C_AvePooling, self).__init__()
        self.ave_real = nn.AvgPool2d(kernel_size, stride=stride, padding=padding,
                                ceil_mode=ceil_mode)
        self.ave_imag = nn.AvgPool2d(kernel_size, stride=stride, padding=padding,
                                     ceil_mode=ceil_mode)
    def forward(self, x):
        return Complex(self.ave_real(x.real),
                       self.ave_imag(x.imag)
                       )
class C_convtranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(C_convtranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight_real = nn.Parameter(
            conv_weight(in_channels, out_channels, kernel_size, stride, padding, transpose=True), requires_grad=True)
        self.weight_imag = nn.Parameter(
            conv_weight(in_channels, out_channels, kernel_size, stride, padding, transpose=True), requires_grad=True)

    def forward(self, complex):
        x_ = F.conv_transpose2d(complex.real, self.weight_real, stride=self.stride, padding=self.padding) - \
             F.conv_transpose2d(complex.imag, self.weight_imag, stride=self.stride, padding=self.padding)
        y_ = F.conv_transpose2d(complex.imag, self.weight_real, stride=self.stride, padding=self.padding) + \
             F.conv_transpose2d(complex.real, self.weight_imag, stride=self.stride, padding=self.padding)
        return Complex(x_, y_)


class C_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(C_conv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.rConv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding)
        self.iConv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding)

    def forward(self, complex):
        r = self.rConv2d(complex.real) - self.iConv2d(complex.imag)
        i = self.rConv2d(complex.imag) + self.iConv2d(complex.real)

        return Complex(r, i)

class C_BatchNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-4, check=False, \
                 momentum=0.1):
        super().__init__()
        self.real_BN = nn.BatchNorm2d(num_features)
        self.im_BN = nn.BatchNorm2d(num_features)
    def forward(self, input):
        return Complex(self.real_BN(input.real), self.im_BN(input.imag))

class C_BatchNorm2d(nn.Module):
    def __init__(self, num_features, affine=True, epsilon=1e-4, check=False, \
                 momentum=0.1, track_running_stats=True):
        super(C_BatchNorm2d, self).__init__()
        self.check = check
        self.affine = affine
        self.epsilon = epsilon
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if self.affine:
            self.bias_real = Parameter(torch.Tensor(num_features), requires_grad=True)
            self.bias_imag = Parameter(torch.Tensor(num_features), requires_grad=True)

            self.gamma_rr = Parameter(torch.Tensor(num_features), requires_grad=True)
            self.gamma_ri = Parameter(torch.Tensor(num_features), requires_grad=True)
            self.gamma_ii = Parameter(torch.Tensor(num_features), requires_grad=True)
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

            self.register_parameter('gamma_rr', None)
            self.register_parameter('gamma_ri', None)
            self.register_parameter('gamma_ii', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))

            self.register_buffer('running_Vrr', torch.ones(num_features) / float(np.sqrt(2.0)))
            self.register_buffer('running_Vii', torch.ones(num_features) / float(np.sqrt(2.0)))
            self.register_buffer('running_Vri', torch.zeros(num_features))

            self.register_buffer('num_batches_tracked', torch.zeros(1))
        else:
            self.register_buffer('running_mean_real', None)
            self.register_buffer('running_mean_imag', None)

            self.register_buffer('running_Vrr', None)
            self.register_buffer('running_Vii', None)
            self.register_buffer('running_Vri', None)

            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean_real.zero_()
            self.running_mean_imag.zero_()

            self.running_Vrr.fill_(1.0 / float(np.sqrt(2.0)))
            self.running_Vii.fill_(1.0 / float(np.sqrt(2.0)))
            self.running_Vri.zero_()

            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant(self.bias_real.data, 0.0)
            nn.init.constant(self.bias_imag.data, 0.0)

            nn.init.constant(self.gamma_rr.data, float(np.sqrt(0.5)))
            nn.init.constant(self.gamma_ri.data, 0)
            nn.init.constant(self.gamma_ii.data, float(np.sqrt(0.5)))

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def forward(self, complex):
        real = complex.real
        imag = complex.imag
        if self.training:
            def mean_along_multiple_dimensions(tensor, dims):
                for i, dim in enumerate(np.sort(dims)):
                    tensor = torch.mean(tensor, dim=int(dim - i))
                return tensor

            real_means = mean_along_multiple_dimensions(real, dims=[0, 2, 3])
            imag_means = mean_along_multiple_dimensions(imag, dims=[0, 2, 3])

            self.running_mean_real = self.running_mean_real * self.momentum + (1.0 - self.momentum) * real_means.data
            self.running_mean_imag = self.running_mean_imag * self.momentum + (1.0 - self.momentum) * imag_means.data

            real_means = real_means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            imag_means = imag_means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            real_centered = real - real_means
            imag_centered = imag - imag_means

            def sum_along_multiple_dimensions(tensor, dims):
                for i, dim in enumerate(np.sort(dims)):
                    tensor = torch.sum(tensor, dim=int(dim - i))
                return tensor

            Vrr = mean_along_multiple_dimensions(real_centered * real_centered, dims=[0, 2, 3]) + self.epsilon
            Vri = mean_along_multiple_dimensions(real_centered * imag_centered, dims=[0, 2, 3])
            Vii = mean_along_multiple_dimensions(imag_centered * imag_centered, dims=[0, 2, 3]) + self.epsilon

            self.running_Vrr = self.running_Vrr * self.momentum + (1.0 - self.momentum) * Vrr.data
            self.running_Vri = self.running_Vri * self.momentum + (1.0 - self.momentum) * Vri.data
            self.running_Vii = self.running_Vii * self.momentum + (1.0 - self.momentum) * Vii.data

            self.num_batches_tracked += 1


        else:
            real_means = Variable(self.running_mean_real.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), requires_grad=False)
            imag_means = Variable(self.running_mean_imag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), requires_grad=False)

            Vrr = Variable(self.running_Vrr, requires_grad=False)
            Vri = Variable(self.running_Vri, requires_grad=False)
            Vii = Variable(self.running_Vii, requires_grad=False)

            real_centered = real - real_means
            imag_centered = imag - imag_means

        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        inverse_st = 1.0 / (float(np.sqrt(2.0)) * s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st

        Wrr = Wrr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        Wri = Wri.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        Wii = Wii.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        real_normed = Wrr * real_centered + Wri * imag_centered
        imag_normed = Wri * real_centered + Wii * imag_centered

        if self.check:
            result_real_means = mean_along_multiple_dimensions(real_normed, dims=[0, 2, 3])
            result_imag_means = mean_along_multiple_dimensions(imag_normed, dims=[0, 2, 3])

            print("real part of result means: ", result_real_means)
            print("imag part of result means: ", result_imag_means)

            Vrr = mean_along_multiple_dimensions(real_normed * real_normed, dims=[0, 2, 3]) + self.epsilon
            Vri = mean_along_multiple_dimensions(real_normed * imag_normed, dims=[0, 2, 3])
            Vii = mean_along_multiple_dimensions(imag_normed * imag_normed, dims=[0, 2, 3]) + self.epsilon

            print('covariance: ', Vrr + Vii)

            print('real part of relation: ', Vrr - Vii)
            print('iamg part of relation: ', 2 * Vri)

        gamma_rr = self.gamma_rr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        gamma_ri = self.gamma_ri.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        gamma_ii = self.gamma_ii.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        bias_real = self.bias_real.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias_imag = self.bias_imag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ans_real = real_normed * gamma_rr + imag_normed * gamma_ri + bias_real
        ans_imag = real_normed * gamma_ri + imag_normed * gamma_ii + bias_imag
        # print(ans_real.shape)
        # print(ans_imag.shape)
        return Complex(ans_real, ans_imag)


class C_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(C_Linear, self).__init__()
        self.rLinear = nn.Linear(in_dim, out_dim)

        self.iLinear = nn.Linear(in_dim, out_dim)

    def forward(self, complex):

        r = self.rLinear(complex.real) - self.iLinear(complex.imag)
        i = self.rLinear(complex.imag) + self.iLinear(complex.real)
        return Complex(r, i)


class C_ReLU(nn.Module):
    def __init__(self):
        super(C_ReLU, self).__init__()

    def forward(self, complex):
        return Complex(F.relu(complex.real), F.relu(complex.imag))


class C_LeakyReLU(nn.Module):
    def __init__(self, alpha=0.001):
        super(C_LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, complex):
        return Complex(F.leaky_relu(complex.real, self.alpha), F.leaky_relu(complex.imag, self.alpha))


class Mod_ReLU(nn.Module):
    def __init__(self, channels):
        super(Mod_ReLU, self).__init__()
        self.b = nn.Parameter(torch.FloatTensor(channels).fill_(0), requires_grad=True)

    def forward(self, complex):
        mag = complex.mag()
        if len(mag.shape) > 2:
            mag = F.relu(mag + self.b[None, :, None, None])
        else:
            mag = F.relu(mag + self.b[None, :])
        res = Complex()
        res.from_polar(mag, complex.phase())
        return res


def complex_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('C_Linear') != -1:
        # real weigths
        fan_in_real, fan_out_real = nn.init._calculate_fan_in_and_fan_out(m.weight_real.data)
        s_real = 1. / (fan_in_real + fan_out_real)  # glorot or xavier criterion
        rng_real = np.random.RandomState(999)
        modulus_real = rng_real.rayleigh(scale=s_real, size=m.weight_real.data.shape)
        phase_real = rng_real.uniform(low=-np.pi, high=np.pi, size=m.weight_real.data.shape)
        weight_real = torch.from_numpy(modulus_real) * torch.cos(torch.from_numpy(phase_real))
        # imag weights
        fan_in_imag, fan_out_imag = nn.init._calculate_fan_in_and_fan_out(m.weight_imag.data)
        s_imag = 1. / (fan_in_imag + fan_out_imag)  # glorot or xavier criterion
        rng_imag = np.random.RandomState(999)
        modulus_imag = rng_imag.rayleigh(scale=s_imag, size=m.weight_imag.data.shape)
        phase_imag = rng_imag.uniform(low=-np.pi, high=np.pi, size=m.weight_imag.data.shape)
        weight_imag = torch.from_numpy(modulus_imag) * torch.cos(torch.from_numpy(phase_imag))

    if classname.find('C_conv2d') != -1:
        # real weigths
        fan_in_real, fan_out_real = nn.init._calculate_fan_in_and_fan_out(m.weight_real.data)
        s_real = 1. / (fan_in_real + fan_out_real)  # glorot or xavier criterion
        rng_real = np.random.RandomState(999)
        modulus_real = rng_real.rayleigh(scale=s_real, size=m.weight_real.data.shape)
        phase_real = rng_real.uniform(low=-np.pi, high=np.pi, size=m.weight_real.data.shape)
        weight_real = torch.from_numpy(modulus_real) * torch.cos(torch.from_numpy(phase_real))
        # imag weights
        fan_in_imag, fan_out_imag = nn.init._calculate_fan_in_and_fan_out(m.weight_imag.data)
        s_imag = 1. / (fan_in_imag + fan_out_imag)  # glorot or xavier criterion
        rng_imag = np.random.RandomState(999)
        modulus_imag = rng_imag.rayleigh(scale=s_imag, size=m.weight_imag.data.shape)
        phase_imag = rng_imag.uniform(low=-np.pi, high=np.pi, size=m.weight_imag.data.shape)
        weight_imag = torch.from_numpy(modulus_imag) * torch.cos(torch.from_numpy(phase_imag))

    if classname.find('C_BatchNorm2d') != -1:
        # real weigths
        fan_in_real, fan_out_real = nn.init._calculate_fan_in_and_fan_out(m.weight_real.data)
        s_real = 1. / (fan_in_real + fan_out_real)  # glorot or xavier criterion
        rng_real = np.random.RandomState(999)
        modulus_real = rng_real.rayleigh(scale=s_real, size=m.weight_real.data.shape)
        phase_real = rng_real.uniform(low=-np.pi, high=np.pi, size=m.weight_real.data.shape)
        weight_real = torch.from_numpy(modulus_real) * torch.cos(torch.from_numpy(phase_real))
        # imag weights
        fan_in_imag, fan_out_imag = nn.init._calculate_fan_in_and_fan_out(m.weight_imag.data)
        s_imag = 1. / (fan_in_imag + fan_out_imag)  # glorot or xavier criterion
        rng_imag = np.random.RandomState(999)
        modulus_imag = rng_imag.rayleigh(scale=s_imag, size=m.weight_imag.data.shape)
        phase_imag = rng_imag.uniform(low=-np.pi, high=np.pi, size=m.weight_imag.data.shape)
        weight_imag = torch.from_numpy(modulus_imag) * torch.cos(torch.from_numpy(phase_imag))


class Sample(nn.Module):
    """
    Foo model
    """

    def __init__(self):
        super(Sample, self).__init__()
        self.conv1 = C_convtranspose2d(3, 3, 3, 1, 1)
        self.relu = C_ReLU()
        self.conv2 = C_convtranspose2d(3, 3, 3, 1, 1)
        self.mod_relu = Mod_ReLU(3)
        self.conv3 = C_convtranspose2d(3, 3, 3, 1, 1)

    def forward(self, complex):
        complex = self.conv1(complex)
        complex = self.relu(complex)
        complex = self.conv2(complex)
        complex = self.mod_relu(complex)
        return self.conv3(complex)