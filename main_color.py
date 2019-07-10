import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DHDN_color import Net
from dataset_color import DatasetFromHdf5
import time, math
import numpy as np
import h5py
from torchsummary import summary

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Densely Connected Hierarchical Network for Image Denoising")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size. Default: 16")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for. Default: 100")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=3,
                    help="Halves the learning rate for every n epochs. Default: n=3")
parser.add_argument("--cuda", action="store_true", help="Use cuda? Default: True")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resume. Default: None")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts). Default: 1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 0")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model. Default: None")
parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument("--checkpoint", default="./checkpoint", type=str,
                    help="Checkpoint path. Default: ./checkpoint ")


def main():
    global opt, model
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("./data/gaus_train_c_50.h5")
    valid_set = "./data/gaus_val_c_50.h5"
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    val = h5py.File(valid_set)
    data_val = val.get('data').value
    label_val = val.get('label').value

    print("===> Building model")
    model = Net()

    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    summary(model, (3, 64, 64))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        avg_loss = train(training_data_loader, optimizer, model, criterion, epoch, data_val, label_val)
        model.eval()
        psnr = 0

        for i in range(data_val.shape[0]):

            val_data = data_val[i, :, :, :]
            val_label = label_val[i, :, :, :]
            val_data = Variable(torch.from_numpy(val_data).float()).view(1, 3, val_data.shape[1], val_data.shape[2])

            if opt.cuda:
                val_data = val_data.cuda()

            with torch.no_grad():
                val_out = model(val_data)
            val_out = val_out.cpu().data[0].numpy()

            psnr += output_psnr_mse(val_label, val_out)

        psnr = psnr / (i + 1)
        save_checkpoint(model, epoch, 99999, psnr, avg_loss)


def adjust_learning_rate(epoch):
    lr = opt.lr
    for i in range(epoch // opt.step):
        lr = lr / 2
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, data_val, label_val):
    lr = adjust_learning_rate(epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()

    max_psnr = 0
    loss_sum = 0
    min_avg_loss = 1
    st = time.time()
    for iteration, batch in enumerate(training_data_loader, 1):

        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        [num_bat, num_c, patch_h, patch_w] = input.shape
        input = input.numpy()
        label = label.numpy()

        a = np.random.randint(4, size=1)[0]
        for i in range(num_bat):
            for j in range(num_c):
                input[i, j, :, :] = np.rot90(input[i, j, :, :], a).copy()
                label[i, j, :, :] = np.rot90(label[i, j, :, :], a).copy()
        if np.random.randint(2, size=1)[0] == 1:
            for i in range(num_bat):
                for j in range(num_c):
                    input[i, j, :, :] = np.flip(input[i, j, :, :], axis=1).copy()
                    label[i, j, :, :] = np.flip(label[i, j, :, :], axis=1).copy()
        if np.random.randint(2, size=1)[0] == 1:
            for i in range(num_bat):
                for j in range(num_c):
                    input[i, j, :, :] = np.flip(input[i, j, :, :], axis=0).copy()
                    label[i, j, :, :] = np.flip(label[i, j, :, :], axis=0).copy()

        input = Variable(torch.from_numpy(input).float()).view(num_bat, num_c, patch_h, patch_w)
        label = Variable(torch.from_numpy(label).float()).view(num_bat, num_c, patch_h, patch_w)

        if opt.cuda:
            input = input.cuda()
            label = label.cuda()

        out = model(input)

        loss = criterion(out, label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        if iteration % 6000 == 0:

            model.eval()
            psnr = 0

            for i in range(data_val.shape[0]):

                val_data = data_val[i, :, :, :]
                val_label = label_val[i, :, :, :]
                val_data = Variable(torch.from_numpy(val_data).float()).view(1, 3, val_data.shape[1], val_data.shape[2])

                if opt.cuda:
                    val_data = val_data.cuda()

                with torch.no_grad():
                    val_out = model(val_data)
                val_out = val_out.cpu().data[0].numpy()

                psnr += output_psnr_mse(val_label, val_out)

            psnr = psnr / (i + 1)
            avg_loss = loss_sum / iteration
            print("===> Epoch[{}]({}/{}): Train_Loss: {:.10f} Val_PSNR: {:.4f}".format(epoch, iteration,
                                                                                       len(training_data_loader),
                                                                                       avg_loss, psnr))
            model.train()

            if psnr > max_psnr or min_avg_loss > avg_loss:
                if psnr > max_psnr:
                    max_psnr = psnr
                if min_avg_loss > avg_loss:
                    min_avg_loss = avg_loss
                save_checkpoint(model, epoch, iteration, psnr, avg_loss)

    print("training_time: ", time.time() - st)
    avg_loss = loss_sum / len(training_data_loader)
    return avg_loss


def save_checkpoint(model, epoch, iteration, psnr, loss):
    model_folder = opt.checkpoint
    model_out_path = model_folder + "/model_epoch_{}_iter_{}_PSNR_{:.4f}_loss_{:.8f}.pth".format(epoch, iteration, psnr,
                                                                                                 loss)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


if __name__ == "__main__":
    main()
