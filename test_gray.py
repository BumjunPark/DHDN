import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time
import scipy.io as sio
import imageio
from myssim import compare_ssim as cal_ssim

parser = argparse.ArgumentParser(description="PyTorch Densely Connected U-Net Test")
parser.add_argument("--cuda", action="store_true", help="Use cuda? Default: True")
parser.add_argument("--ens", action="store_false", help="Model ensemble? Default: False")
parser.add_argument("--model1", default="./retrain/gaus_g_50.pth", type=str, help="Model path")
parser.add_argument("--model2", default="", type=str, help="Model path for model ensemble")
parser.add_argument("--data", default="./data/noisy_kodak_nl50_g.mat", type=str, help="Test data path")
parser.add_argument("--gt", default="./data/gt_kodak_g.mat", type=str, help="GT data path")
parser.add_argument("--gpu", default='0', help="GPU number to use when testing. Default: 0")
parser.add_argument("--result", default="./result/", type=str, help="Result path Default: ./result/")

opt = parser.parse_args()

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model1 = torch.load(opt.model1, map_location=lambda storage, loc: storage)["model"]
if ens:
    model2 = torch.load(opt.model2, map_location=lambda storage, loc: storage)["model"]

test_path = opt.data
gt_path = opt.gt

avg_ssim_noisy = 0.0
avg_ssim_predicted = 0.0
avg_ssim_ens_predicted = 0.0
avg_psnr_noisy = 0.0
avg_psnr_predicted = 0.0
avg_psnr_ens_predicted = 0.0
avg_elapsed_time = 0.0
count = 0.0
flag = 0

if ens:
    avg_ssim_predicted2 = 0.0
    avg_ssim_ens_predicted2 = 0.0
    avg_ssim_m_ens_predicted = 0.0
    avg_psnr_predicted2 = 0.0
    avg_psnr_ens_predicted2 = 0.0
    avg_psnr_m_ens_predicted = 0.0

raw_val_data = sio.loadmat(test_path)['data_val']
raw_val_data = raw_val_data.astype(float)
gt = sio.loadmat(gt_path)['label_val']
gt = gt.astype(float)
if raw_val_data.shape[0] % 8 != 0 or raw_val_data.shape[1] % 8 != 0:
    flag = 1
    h = 8 - (raw_val_data.shape[0] % 8)
    w = 8 - (raw_val_data.shape[1] % 8)
    pad_raw_val_data = np.zeros((raw_val_data.shape[0] + h, raw_val_data.shape[1] + w, raw_val_data.shape[2], raw_val_data.shape[3]))
    [patch_h, patch_w, c, num_pic] = pad_raw_val_data.shape  # h w c pic
    for a in range(raw_val_data.shape[2]):
        for b in range(raw_val_data.shape[3]):
            pad_raw_val_data[:, :, a, b] = np.pad(raw_val_data[:, :, a, b], ((0, h), (0, w)), 'constant', constant_values=(0))

else:
    [patch_h, patch_w, c, num_pic] = raw_val_data.shape #h w c pic

for i in range(num_pic):

    count += 1
    val_data = []
    out_data = []
    temp1 = np.zeros((c, patch_w, patch_h))
    temp2 = np.zeros((c, patch_h, patch_w))
    temp3 = np.zeros((c, patch_w, patch_h))
    noisy = np.zeros((c, patch_h, patch_w))
    ens_output = np.zeros((c, patch_h, patch_w))

    if ens:
        out_data2 = []
        ens_output2 = np.zeros((c, patch_h, patch_w))
        m_ens_output = np.zeros((c, patch_h, patch_w))

    for k in range(c):
        if flag:
            noisy[k, :, :] = pad_raw_val_data[:, :, k, i]

        else:
            noisy[k, :, :] = raw_val_data[:, :, k, i]  # val_data (3,256,256)
    val_data.append(noisy)

    val_data.append(np.fliplr(noisy).copy())

    for a in range(c):
        temp1[a, :, :] = np.rot90(noisy[a, :, :], 1)
    val_data.append(temp1)

    val_data.append(np.fliplr(val_data[2]).copy())

    for a in range(c):
        temp2[a, :, :] = np.rot90(noisy[a, :, :], 2)
    val_data.append(temp2)

    val_data.append(np.fliplr(val_data[4]).copy())

    for a in range(c):
        temp3[a, :, :] = np.rot90(noisy[a, :, :], 3)
    val_data.append(temp3)

    val_data.append(np.fliplr(val_data[6]).copy())

    for a in range(8):
        [tc, th, tw] = val_data[a].shape
        input = Variable(torch.from_numpy(val_data[a]).float()).view(1, tc, th, tw)

        if cuda:
            model1 = model1.cuda()
            input = input.cuda()

            if ens:
                model2 = model2.cuda()

        start_time = time.time()
        with torch.no_grad():
            output = model1(input)
            output = output.cpu()
            out_data.append(output.detach().numpy().astype(np.float32))
            if ens:
                output2 = model2(input)
                output2 = output2.cpu()
                out_data2.append(output2.detach().numpy().astype(np.float32))

        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

    results = np.zeros((8, c, patch_h, patch_w))

    results[0, :, :, :] = out_data[0][0, :, :, :]
    results[1, :, :, :] = np.fliplr(out_data[1][0, :, :, :])
    temp1 = np.fliplr(out_data[3][0, :, :, :]).copy()
    temp2 = np.fliplr(out_data[5][0, :, :, :]).copy()
    temp3 = np.fliplr(out_data[7][0, :, :, :]).copy()
    for a in range(c):
        results[2, a, :, :] = np.rot90(out_data[2][0, a, :, :], 3)
        results[3, a, :, :] = np.rot90(temp1[a, :, :], 3)
        results[4, a, :, :] = np.rot90(out_data[4][0, a, :, :], 2)
        results[5, a, :, :] = np.rot90(temp2[a, :, :], 2)
        results[6, a, :, :] = np.rot90(out_data[6][0, a, :, :], 1)
        results[7, a, :, :] = np.rot90(temp3[a, :, :], 1)

    if ens:
        results2 = np.zeros((8, c, patch_h, patch_w))

        results2[0, :, :, :] = out_data2[0][0, :, :, :]
        results2[1, :, :, :] = np.fliplr(out_data2[1][0, :, :, :])
        temp1 = np.fliplr(out_data2[3][0, :, :, :]).copy()
        temp2 = np.fliplr(out_data2[5][0, :, :, :]).copy()
        temp3 = np.fliplr(out_data2[7][0, :, :, :]).copy()
        for a in range(c):
            results2[2, a, :, :] = np.rot90(out_data2[2][0, a, :, :], 3)
            results2[3, a, :, :] = np.rot90(temp1[a, :, :], 3)
            results2[4, a, :, :] = np.rot90(out_data2[4][0, a, :, :], 2)
            results2[5, a, :, :] = np.rot90(temp2[a, :, :], 2)
            results2[6, a, :, :] = np.rot90(out_data2[6][0, a, :, :], 1)
            results2[7, a, :, :] = np.rot90(temp3[a, :, :], 1)

    for a in range(8):
        ens_output += results[a, :, :, :]
        if ens:
            ens_output2 += results2[a, :, :, :]
            m_ens_output += results[a, :, :, :] + results2[a, :, :, :]

    out = results[0, :, :, :]
    out[out < 0] = 0
    out[out > 1] = 1.0
    ens_output /= 8.0
    ens_output[ens_output < 0] = 0
    ens_output[ens_output > 1] = 1.0
    if ens:
        out2 = results2[0, :, :, :]
        out2[out2 < 0] = 0
        out2[out2 > 1] = 1.0
        ens_output2 /= 8.0
        ens_output2[ens_output2 < 0] = 0
        ens_output2[ens_output2 > 1] = 1.0
        m_ens_output /= 16.0
        m_ens_output[m_ens_output < 0] = 0
        m_ens_output[m_ens_output > 1] = 1.0

    if flag:
        patch_h = patch_h - h
        patch_w = patch_w - w
    temp = np.zeros((patch_h, patch_w, c))
    ens_temp = np.zeros((patch_h, patch_w, c))
    noisy_temp = np.zeros((patch_h, patch_w, c))
    if ens:
        temp2 = np.zeros((patch_h, patch_w, c))
        ens_temp2 = np.zeros((patch_h, patch_w, c))
        m_ens_temp = np.zeros((patch_h, patch_w, c))
    for u in range(patch_h):
        for v in range(patch_w):
            temp[u, v, :] = out[:, u, v]
            ens_temp[u, v, :] = ens_output[:, u, v]
            noisy_temp[u, v, :] = noisy[:, u, v]
            if ens:
                temp2[u, v, :] = out2[:, u, v]
                ens_temp2[u, v, :] = ens_output2[:, u, v]
                m_ens_temp[u, v, :] = m_ens_output[:, u, v]

    original = np.zeros((patch_h, patch_w, c))
    original[:, :, :] = gt[:, :, :, i]
    print("image number : {}".format(i))
    psnr_noisy = output_psnr_mse(noisy_temp, original)
    print("psnr_noisy : {}".format(psnr_noisy))
    avg_psnr_noisy += psnr_noisy
    psnr_predicted = output_psnr_mse(temp, original)
    print("psnr_predicted : {}".format(psnr_predicted))
    avg_psnr_predicted += psnr_predicted
    psnr_ens_predicted = output_psnr_mse(ens_temp, original)
    print("psnr_ens_predicted : {}".format(psnr_ens_predicted))
    avg_psnr_ens_predicted += psnr_ens_predicted
    ssim_noisy = cal_ssim(noisy_temp, original, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
    print("ssim_noisy : {}".format(ssim_noisy))
    avg_ssim_noisy += ssim_noisy
    ssim_predicted = cal_ssim(temp, original, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
    print("ssim_predicted : {}".format(ssim_predicted))
    avg_ssim_predicted += ssim_predicted
    ssim_ens_predicted = cal_ssim(ens_temp, original, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
    print("ssim_ens_predicted : {}".format(ssim_ens_predicted))
    avg_ssim_ens_predicted += ssim_ens_predicted
    if ens:
        psnr_predicted2 = output_psnr_mse(temp2, original)
        print("psnr_predicted2 : {}".format(psnr_predicted2))
        avg_psnr_predicted2 += psnr_predicted2
        psnr_ens_predicted2 = output_psnr_mse(ens_temp2, original)
        print("psnr_ens_predicted2 : {}".format(psnr_ens_predicted2))
        avg_psnr_ens_predicted2 += psnr_ens_predicted2
        psnr_m_ens_predicted = output_psnr_mse(m_ens_temp, original)
        print("psnr_m_ens_predicted : {}".format(psnr_m_ens_predicted))
        avg_psnr_m_ens_predicted += psnr_m_ens_predicted
        ssim_predicted2 = cal_ssim(temp2, original, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
        print("ssim_predicted2 : {}".format(ssim_predicted2))
        avg_ssim_predicted2 += ssim_predicted2
        ssim_ens_predicted2 = cal_ssim(ens_temp2, original, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
        print("ssim_ens_predicted2 : {}".format(ssim_ens_predicted2))
        avg_ssim_ens_predicted2 += ssim_ens_predicted2
        ssim_m_ens_predicted = cal_ssim(m_ens_temp, original, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
        print("ssim_m_ens_predicted : {}".format(ssim_m_ens_predicted))
        avg_ssim_m_ens_predicted += ssim_m_ens_predicted

    name = str(i) + '.png'
    temp = temp * 255.0
    temp = np.uint8(np.round(temp))
    imageio.imwrite(opt.result + name, temp)
    name = str(i) + '_ens.png'
    ens_temp = ens_temp * 255.0
    ens_temp = np.uint8(np.round(ens_temp))
    imageio.imwrite(opt.result + name, ens_temp)
    if ens:
        name = str(i) + '_2.png'
        temp2 = temp2 * 255.0
        temp2 = np.uint8(np.round(temp2))
        imageio.imwrite(opt.result + name, temp2)
        name = str(i) + '_2_ens.png'
        ens_temp2 = ens_temp2 * 255.0
        ens_temp2 = np.uint8(np.round(ens_temp2))
        imageio.imwrite(opt.result + name, ens_temp2)
        name = str(i) + '_m_ens.png'
        m_ens_temp = m_ens_temp * 255.0
        m_ens_temp = np.uint8(np.round(m_ens_temp))
        imageio.imwrite(opt.result + name, m_ens_temp)

time_per_patch = avg_elapsed_time / (count * 8)
avg_psnr_noisy /= count
avg_psnr_predicted /= count
avg_psnr_ens_predicted /= count
avg_ssim_noisy /= count
avg_ssim_predicted /= count
avg_ssim_ens_predicted /= count
print("avg_psnr_noisy : {}".format(avg_psnr_noisy))
print("avg_psnr_predicted : {}".format(avg_psnr_predicted))
print("avg_psnr_ens_predicted : {}".format(avg_psnr_ens_predicted))
print("avg_ssim_noisy : {}".format(avg_ssim_noisy))
print("avg_ssim_predicted : {}".format(avg_ssim_predicted))
print("avg_ssim_ens_predicted : {}".format(avg_ssim_ens_predicted))
if ens:
    time_per_patch /= 2.0
    avg_psnr_predicted2 /= count
    avg_psnr_ens_predicted2 /= count
    avg_psnr_m_ens_predicted /= count
    avg_ssim_predicted2 /= count
    avg_ssim_ens_predicted2 /= count
    avg_ssim_m_ens_predicted /= count
    print("avg_psnr_predicted2 : {}".format(avg_psnr_predicted2))
    print("avg_psnr_ens_predicted2 : {}".format(avg_psnr_ens_predicted2))
    print("avg_psnr_m_ens_predicted : {}".format(avg_psnr_m_ens_predicted))
    print("avg_ssim_predicted2 : {}".format(avg_ssim_predicted2))
    print("avg_ssim_ens_predicted2 : {}".format(avg_ssim_ens_predicted2))
    print("avg_ssim_m_ens_predicted : {}".format(avg_ssim_m_ens_predicted))

print("run time for mega pixel: {}".format(time_per_patch * 1000000 / (patch_h * patch_w)))


