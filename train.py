"""CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --width norm
in oder to use VM i remove the /255"""

"""20201022 use 5*5 conv instead of the 3*3 deliation conv lr=0.0001 loss=KL"""
"""CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
in oder to use VM i remove the /255"""
#OAS1做训练集，OAS2做测试集。采用最邻近插值进行resize：*****结果不错
import warnings
import torch
from losses import jacobian_determinant1
import torch.nn as nn
import shutil
from torch.optim import Adam, SGD
from DataLoad import Dataset_epoch
import argparse
import os
import time
import numpy as np
import losses
from model import Net
from Validation import validation
from torch.optim import lr_scheduler
import csv
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import torch.utils.data as Data
import sys

sys.path.append("/home/songlei/train/ext/neuron/")
sys.path.append("/home/songlei/train/ext/pynd-lib/")
sys.path.append("/home/songlei/train/ext/pytools-lib/")
sys.path.append("/home/songlei/train/ext/medipy-lib/")



import glob

warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser(description='param')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--iters', default=150000, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--loss_name', default='MSE', type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--reg_param', default=0.02,
                    type=float)  # NCC 1 MSE 0.01
parser.add_argument('--path', default="/home/songlei/OASdata/OAS_new/train/", type=str)
parser.add_argument('--atlas_file', default="/home/songlei/OASdata/OAS_new/atlases/", type=str)#UM_379
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--checkpoint_path',
                    default='/home/newdisk/songlei/train/version_ResT_new/ResT1/checkpoint&log_token/',
                    type=str)
parser.add_argument('--early_stop', default=False, type=bool)
parser.add_argument('--model', default='mymodel', type=str)
parser.add_argument('--width', default='norm', type=str)
parser.add_argument('--json_file', default="OAS.json")
parser.add_argument('--test_file', default="/home/songlei/OASdata/OAS_new/valsets/", type=str)
parser.add_argument('--log_folder', default='/home/newdisk/songlei/train/version_ResT_new/ResT1/checkpoint_token/', type=str)
parser.add_argument('--init_params', default=False, type=bool)
# parser.add_argument('--save_loss_path', default='./', type=str)

args = parser.parse_args()
CUDA_VISIBLE_DEVICES=1
torch.backends.cudnn.benchmark = True  # use cudnn training time reduced

device1 = torch.device('cuda:{}'.format('1') if torch.cuda.is_available() else 'cpu')
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


def Train(iters,
          batch_size,
          loss_name,
          lr,
          reg_param,
          train_path,
          atlas_file,
          resume,#???
          json_file,
          bn,
          checkpoint_path
          ):

    print(train_path)
    encoder= [16, 32, 32, 32]
    decoder=[32, 32, 32, 32, 32, 16, 16]#[32, 32, 32, 32, 8, 8]
    # init the tensorboardX
    writer = SummaryWriter(args.log_folder + f'ls{loss_name}lr{lr}/' + f's{reg_param}_bn_{bn}/')

    reg_param = reg_param

    model = Net((1,1,96, 112 ,96), 3).apply(initialize_weights).cuda(device=device1)

    # loss_fun = losses.KL_Divergence
    if loss_name == 'MSE':
        loss_fun=losses.MSE().loss
    elif loss_name == 'NCC':
        loss_fun=losses.NCC()

    else:
        raise Exception("MMR: Loss function must NCC or MSE")
    Grad_loss = losses.Grad().loss
    opt = Adam(model.parameters(), lr=lr)
    #if koad model
    if resume:
        counter = 0
        flag = 0
        check_point = torch.load("/home/newdisk/songlei/train/version_ResT_new/ResT1/checkpoint&log_up/BN_False/lsMSElr0.0001/s0.02/norm_checkpoint.pth.tar", map_location='cpu')
        # print(f'Checkpoint Keys : {check_point.keys()}.')
        current_iter = check_point['iter_th'] + 1

        best_acc1 = check_point['best_acc']
        flag = best_acc1
        print(f'Training restart at : {current_iter}th epoch.', flush=True)

        model.load_state_dict(check_point['state_dict'])
        opt.load_state_dict(check_point['optimizer'])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device=device1)

    else:
        flag = 0
        current_iter = 0
    #djusting Learning Rate
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5, patience=5, verbose=False,threshold=0.00001, min_lr=0.00001, cooldown=10)
    #Generate training data
    names = sorted(glob.glob(train_path + '/*.npy'))
    trainset_loader = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=2)

    data_size = len(trainset_loader)
    print("Data size is {}. ".format(data_size))
    atlas_path = atlas_file
    val_path = args.test_file


    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

    while current_iter <= iters:
        main_time = time.time()
        tmp_a = 0.0
        tmp_b = 0.0
        tmp_c = 0.0
        loss_iters = 0.0

        start_time = time.time()

        loss_epoch = 0.0

        # scheduler sleep
        if current_iter >= 12000:

            scheduler_start = True
        else:
            scheduler_start = False

        for X, Y in tqdm(trainset_loader):
            """
            X :moving
            Y: fixed
            """
            X_cuda = X.cuda(device=device1)
            Y_cuda = Y.cuda(device=device1)

            X_Y, X_Y_flow = model(X_cuda, Y_cuda)

            # X->Y
            flow_per = X_Y_flow.permute(0, 2, 3, 4, 1)
            flow_per = flow_per.squeeze().detach().cpu()
            jac = jacobian_determinant1(flow_per)
            jac_neg_per = np.sum([i < 0 for i in jac]) / (
                    jac.shape[1] * jac.shape[2] *  jac.shape[0])
            tmp_c = tmp_c+jac_neg_per

            loss_a = loss_fun(Y_cuda, X_Y)
            loss_b = Grad_loss(X_Y_flow)
            loss = loss_a + reg_param * loss_b

            tmp_a = tmp_a + loss_a.item()
            tmp_b = tmp_b + loss_b.item()
            loss_iters = loss_iters + loss.item()


            opt.zero_grad()
            loss.backward()
            opt.step()

            # tensorboard record
            if current_iter % 400 == 0:

                writer.add_scalars(f'{loss_name}_loss', {f'{loss_name}_loss': tmp_a / 400},
                                   current_iter)
                writer.add_scalars('smooth_loss', {'smooth_loss': tmp_b / 400},
                                   current_iter)

                writer.add_scalars('nej', {'nej': tmp_c / 400}, current_iter)
                # writer.add_scalars('nej_inv', {'nej': tmp_d / 400}, current_iter)

                writer.add_scalars('epoch_loss', {'epoch_loss': loss_iters / 400}, current_iter)
                writer.close()
                tmp_a = 0.0
                tmp_b = 0.0
                tmp_c = 0.0
                loss_iters = 0.0

            # tensorboard for viewing
            if current_iter % 1200 == 0:
                acc, val_time, atlas_slice, volume_slice, pred_slice, jac_det_slice, flow, jac_det_neg_per, atlas_label_slice, pred_label_slice = \
                    validation(atlas_file=atlas_path , val_file=val_path, acc_fn=losses.dice, model=model, labels=labels, slice=56)

                fig = show(atlas_slice, volume_slice, pred_slice, jac_det_slice)
                import sys

                sys.path.append("/home/songlei/train/ext/neuron/")
                sys.path.append("/home/songlei/train/ext/pynd-lib/")
                sys.path.append("/home/songlei/train/ext/pytools-lib/")
                sys.path.append("/home/songlei/train/ext/medipy-lib/")
                import neuron

                flow_2D = flow[:, :, :, 56, :]
                plate = torch.zeros(1, 2, 96, 96)

                # need to find a correct direction for viewing.
                plate[:, 0, :, :] = flow_2D[:, 0, :, :]
                plate[:, 1, :, :] = flow_2D[:, 1, :, :]

                plate = plate.permute(0, 2, 3, 1).cpu().detach().numpy()
                flow_fig, axs = neuron.plot.flow([plate.squeeze()[::2, ::2]], width=5,
                                                 show=False)  # [::2,::2] is removable.

                writer.add_scalars('dice score', {'dice_score': acc}, current_iter)
                writer.add_figure('Validation', fig, current_iter)
                writer.add_figure('flow direction 2D', flow_fig, current_iter)
                writer.add_scalars('jac_det negative percent', {'percent': jac_det_neg_per}, current_iter)
                writer.close()

                if scheduler_start:
                    scheduler.step(acc)

                print(f"Iter:{current_iter}th. Present LR:{opt.state_dict()['param_groups'][0]['lr']}.")

                if flag < acc:
                    is_best = True
                    update = 'True'
                    save_checkpoint({'iter_th': current_iter, 'loss': loss_iters,
                                     'state_dict': model.state_dict(), 'best_acc': acc,
                                     'optimizer': opt.state_dict(), },
                                    is_best, checkpoint_path)
                    flag = acc
                else:
                    update = ' '

                print(''.center(80, '='), flush=True)
                print("\t\titers: {}".format(current_iter), flush=True)
                print("\t\tLoss: {}".format(loss), flush=True)
                print("\t\tAccuracy (Dice score): {}.".format(acc), flush=True)
                print("\t\tJAC (Train): {}.".format(jac_neg_per), flush=True)
                print("\t\tJAC (Validation): {}.".format(jac_det_neg_per), flush=True)
                print("\t\tValidation time spend: {:.2f}s".format(val_time), flush=True)
                print(''.center(80, '='), flush=True)

                if not os.path.exists(checkpoint_path + f'{reg_param}' + '_log.csv'):
                    with open(checkpoint_path + f'{reg_param}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = ['iter_th', 'LR', 'loss', 'validation', "JAC", 'update', 'lr']
                        csv_write.writerow(row)
                else:
                    with open(checkpoint_path + f'{reg_param}' + '_log.csv', 'a') as f:
                        csv_write = csv.writer(f)
                        row = [current_iter, opt.state_dict()['param_groups'][0]['lr'], loss_iters,
                               acc, jac_det_neg_per, update, lr]
                        csv_write.writerow(row)

            # save checkpoint
            if current_iter % 3000 == 0:
                save_checkpoint({'iter_th': current_iter, 'loss': loss_iters, 'state_dict': model.state_dict(),
                                 'best_acc': flag, 'optimizer': opt.state_dict(), }, is_best=False,
                                checkpoint_path=checkpoint_path, filename=f'{current_iter}_checkpoint.pth.tar')
            current_iter += 1
            if current_iter > iters:
                break



def Run_Distribution():
    for lr in [0.0001]:
        reg_param_list = [args.reg_param]
        for i in reg_param_list:
            bn=False

            checkpoint_path = args.checkpoint_path +f'BN_{bn}/' +f'ls{args.loss_name}lr{lr}/' + f's{i}/'
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            print(f"Now, this experiment's parameters [Learn Rate = {lr}] [Regression para = {i}] [BN = {bn}]")
            Train(iters=args.iters,
                  batch_size=args.batch_size,
                  loss_name=args.loss_name,
                  lr=lr,
                  reg_param=args.reg_param,
                  train_path=args.path,
                  atlas_file=args.atlas_file,
                  resume=args.resume,
                  json_file=args.json_file,bn=bn,checkpoint_path=checkpoint_path)


def early_stop(epoch, counter, val_acc, threshold, flag, resumed_epoch):
    if counter < threshold:
        if val_acc >= flag:
            mark = val_acc
            mode = 'update'
            shutdown = False
            if args.local_rank == 0:
                print(f"--------epoch={epoch}, mode={mode}, mark={mark}------------------------------\n", flush=True)
            return mark, mode, shutdown

        else:
            mark = flag
            mode = 'hold'
            shutdown = False
            if args.local_rank == 0:
                print(f"--------epoch={epoch}, mode={mode}, mark={mark}------------------------------\n", flush=True)
            return mark, mode, shutdown

    else:
        if counter >= threshold:
            if args.local_rank == 0:
                print(f"Mission stopped via early stop, the end epoch is {epoch}", flush=True)
            mark = None
            mode = None,
            shutdown = True
            return mark, mode, shutdown


def save_checkpoint(state, is_best, checkpoint_path,filename='checkpoint.pth.tar'):
    if args.model == 'VM':
        if args.local_rank == 0:

            print("loaded save_checkpoint fun")
            torch.save(state, checkpoint_path + 'VM_' + filename)
            if is_best:
                shutil.copyfile(checkpoint_path + 'VM_' + filename,
                                checkpoint_path + 'VM_' + 'model_best.pth.tar')
            #     print("updata the paras file !")
            # print("success save_checkpoint fun")
    else:
        if args.local_rank == 0:
            best_val = []
            best_val.append(state['best_acc'])
            # print("loaded save_checkpoint fun")
            torch.save(state, checkpoint_path + f'{args.width}_' + filename)
            if is_best:
                shutil.copyfile(checkpoint_path + f'{args.width}_' + filename,
                                checkpoint_path + f'{args.width}_' + 'model_best.pth.tar')
                print('\tAccuracy is updated and the params is saved in [model_best.pth.tar]!'.ljust(20), flush=True)
            if state['iter_th'] == args.iters:
                os.rename(checkpoint_path + f'{args.width}_' + 'model_best.pth.tar',
                          checkpoint_path + f'{max(best_val)}_' + 'model_best.pth.tar')


def HistNorm(img, low_bound=0, high_bound=255):
    # used in torch
    # scale data to [low_bound,high_bound]
    img_max = img.max()
    img_min = img.min()
    out = (high_bound - low_bound) / (img_max - img_min) * (img - img_min) + low_bound
    out[out < low_bound] = low_bound
    out[out > high_bound] = high_bound
    return out


def slice_histNorm(img, low_bound=0, high_bound=255):
    """

    :param img:fixed,[D,H,W] moving[B,C,D,H,W]
    :param low_bound:
    :param high_bound:
    :return:
    """

    if len(img.shape) == 5:
        slice_list = []
        stride = img.shape[2]
        for i in range(stride):
            img_max = img[:, :, i, :, :].max()
            img_min = img[:, :, i, :, :].min()
            out = (high_bound - low_bound) / (img_max - img_min) * (img[:, :, i, :, :] - img_min) + low_bound
            out[out < low_bound] = low_bound
            out[out > high_bound] = high_bound
            slice_list.append(out)

        for i in range(stride):
            img = torch.cat([slice_list[i], slice_list[i + 1]])
    else:
        slice_list = []
        for i in range(img.shape[0]):
            img_max = img[i, :, :].max()
            img_min = img[i, :, :].min()
            out = (high_bound - low_bound) / (img_max - img_min) * (img[i, :, :] - img_min) + low_bound
            out[out < low_bound] = low_bound
            out[out > high_bound] = high_bound
            slice_list.append(out)


def show(atlas, img, pred, jac_det):
    fig, ax = plt.subplots(1, 4)
    fig.dpi = 200

    ax0 = ax[0].imshow(atlas, cmap='gray')
    ax[0].set_title('atlas')
    ax[0].axis('off')
    cb0 = fig.colorbar(ax0, ax=ax[0], shrink=0.2)
    cb0.ax.tick_params(labelsize='small')

    ax1 = ax[1].imshow(img, cmap='gray')
    ax[1].set_title('moving')
    ax[1].axis('off')
    cb1 = fig.colorbar(ax1, ax=ax[1], shrink=0.2)
    cb1.ax.tick_params(labelsize='small')

    ax2 = ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('pred')
    ax[2].axis('off')
    cb2 = fig.colorbar(ax2, ax=ax[2], shrink=0.2)
    cb2.ax.tick_params(labelsize='small')

    ax3 = ax[3].imshow(jac_det, cmap='bwr', norm=MidpointNormalize(midpoint=1))
    ax[3].set_title('jac_det')
    ax[3].axis('off')
    cb3 = fig.colorbar(ax3, ax=ax[3], shrink=0.2)
    cb3.ax.tick_params(labelsize='small')
    return fig


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def MinMaxNorm(img, use_gpu=False):
    if use_gpu:
        Max = torch.max(img)
        Min = torch.min(img)
        return (img - Min) / (Max - Min)
    else:
        Max = img.max()
        Min = img.min()
        return (img - Min) / (Max - Min)


if __name__ == '__main__':
    Run_Distribution()

# train(500,1000,'KL')
