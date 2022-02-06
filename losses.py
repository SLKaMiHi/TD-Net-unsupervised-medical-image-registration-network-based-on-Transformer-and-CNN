import torch
import torch.nn.functional as F
import numpy as np
import math


def gradient_loss(s, penalty='l1'):
    dy = torch.abs(s[:, :, 1:, :,:] - s[:, :, :-1, :,:])
    dx = torch.abs(s[:, :, :, 1:,:] - s[:, :, :, :-1,:])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy)+ torch.mean(dz)
    return d / 3.0
# def gradient_loss(s, penalty='l2'):
#     dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
#     dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
#
#     if (penalty == 'l2'):
#         dy = dy * dy
#         dx = dx * dx
#
#     d = torch.mean(dx) + torch.mean(dy)
#     return d / 2.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 * torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice

def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    if ndims==2:
        I_var, J_var, cross = compute_local_sums_2d(I, J, sum_filt, stride, padding, win)

    else:
        I_var, J_var, cross = compute_local_sums_3d(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)
    # cc = (cross + 1e-5) ** 2 / ((I_var + 1e-5) * (J_var + 1e-5)) # improved

    return -1 * torch.mean(cc)


def compute_local_sums_3d(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def compute_local_sums_2d(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def KL_Divergence(warped, fixed):
    # print(f'warp:{warped.shape}')
    # print(f'fixed:{fixed.shape}')
    warped=warped.contiguous().view(1,-1)
    fixed = fixed.contiguous().view(1, -1)
    warped_softmax=warped.softmax(1)
    fixed_softmax=fixed.softmax(1)
    KL_Div = F.kl_div(fixed_softmax.log(), warped_softmax,reduction='sum')
    return KL_Div


def dice_score(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    top = 2 * torch.sum(pred * target, [1, 2, 3])
    union = torch.sum(pred + target, [1, 2, 3])
    eps = torch.ones_like(union) * 1e-5
    bottom = torch.max(union, eps)
    dice = torch.mean(top / bottom)
    #print("Dice score", dice)
    return dice

def compute_label_dice(src, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [ 2,4,8,10,11,12,13,16,17,18,24,26,28,
                47,49,50,51,52,58,60,   ]
    '''cls_lst =[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 181, 182]'''
    dice_lst = []
    for cls in cls_lst:
        dice = DSC(src == cls, pred == cls)
        dice_lst.append(dice)
    aa = np.array(dice_lst)


    return np.mean(dice_lst),aa
def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return np.mean(dicem)

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# def jacobian_determinant(disp):
#     """
#     jacobian determinant of a displacement field.
#     NB: to compute the spatial gradients, we use np.gradient.
#     Parameters:
#         disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
#               where vol_shape is of len nb_dims
#     Returns:
#         jacobian determinant (scalar)
#     """
#     def ndgrid(*args, **kwargs):
#         """
#         Disclaimer: This code is taken directly from the scitools package [1]
#         Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
#         To avoid issues, we copy the quick code here.
#         Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
#         ``meshgrid`` for documentation).
#         """
#         kwargs['indexing'] = 'ij'
#         return np.meshgrid(*args, **kwargs)
#
#     def volsize2ndgrid(volsize):
#         """
#         return the dense nd-grid for the volume with size volsize
#         essentially return the ndgrid fpr
#         """
#         ranges = [np.arange(e) for e in volsize]
#         return ndgrid(*ranges)
#
#     # check inputs
#     volshape = disp.shape[:-1]
#     nb_dims = len(volshape)
#     assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
#
#     # compute grid
#
#     # grid_lst = nd.volsize2ndgrid(volshape)
#
#     ranges = [np.arange(e) for e in volshape]
#     grid_lst=np.meshgrid(ranges)
#
#     grid = np.stack(grid_lst[0], len(volshape))
#
#     # compute gradients
#     J = np.gradient(disp + grid)
#
#     # 3D glow
#     if nb_dims == 3:
#         dx = J[0]
#         dy = J[1]
#         dz = J[2]
#
#         # compute jacobian components
#         Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
#         Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
#         Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
#
#         return Jdet0 - Jdet1 + Jdet2
#
#     else: # must be 2
#
#         dfdx = J[0]
#         dfdy = J[1]
#
#         return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def Get_Jac(displacement):
    '''
    the expected input: displacement of shape(batch, H, W, D, channel),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*((D_y[...,1]+1)*(D_z[...,2]+1) - D_y[...,2]*D_z[...,1])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    D = D1 - D2 + D3

    return D
import pystrum.pynd.ndutils as nd
def jacobian_determinant(flow):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    JAC = []
    for i in range(0,10):
        disp = flow[i]
        volshape = disp.shape[:-1]
        nb_dims = len(volshape)
        assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

        # compute grid
        grid_lst = nd.volsize2ndgrid(volshape)
        grid = np.stack(grid_lst, len(volshape))

        # compute gradients
        J = np.gradient(disp + grid)

        # 3D glow
        if nb_dims == 3:
            dx = J[0]
            dy = J[1]
            dz = J[2]

            # compute jacobian components
            Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
            Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
            Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

            JAC.append(Jdet0 - Jdet1 + Jdet2)
    kk = np.array(JAC)
    return kk
def jacobian_determinant1(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else: # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

"""new version of loss"""
import torch
import torch.nn.functional as F
import numpy as np
import math

device1 = torch.device('cuda:{}'.format('1') if torch.cuda.is_available() else 'cpu')
class NCC_old:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device1)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)
class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
