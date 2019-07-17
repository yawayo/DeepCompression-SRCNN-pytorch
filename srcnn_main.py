import argparse
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srcnn_data import get_training_set, get_test_set
from srcnn_model import SRCNN
import util

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--log', type=str, default='log.txt', help='log file name')
parser.add_argument('--gpuids', default=[0, 1, 2, 3], nargs='+', help='GPU ID for using')
parser.add_argument('--sensitivity', type=float, default=2, help="sensitivity value that is multiplied to layer's std in order to get threshold value")
opt = parser.parse_args()

print(opt)

use_cuda = opt.cuda

torch.manual_seed(opt.seed)
if use_cuda:
    torch.cuda.manual_seed(opt.seed)

train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

srcnn = SRCNN(mask=True)
util.print_model_parameters(srcnn)

criterion = nn.MSELoss()

if opt.cuda:
    torch.cuda.set_device(opt.gpuids[0])
    with torch.cuda.device(opt.gpuids[0]):
        srcnn = srcnn.cuda()
        criterion = criterion.cuda()
    #srcnn = nn.DataParallel(srcnn, device_ids=opt.gpuids, output_device=opt.gpuids[0])

optimizer = optim.Adam(srcnn.parameters(),lr=opt.lr)
initial_optimizer_state_dict = optimizer.state_dict()


def train(epoch):
    epoch_loss = 0
    pre = []
    post = []
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        model_out = srcnn(input)
        loss = criterion(model_out, target)
        epoch_loss += loss.item()
        loss.backward()

        for name, p in srcnn.named_parameters():
            if 'mask' in name:
                continue
            if 'label' in name:
                continue
            for param_group in optimizer.param_groups:
                temp = param_group['lr']
            tensor = p.data.cpu().numpy()
            temp_t = np.copy(tensor)
            grad_tensor = p.grad.data.cpu().numpy()
            grad_tensor = np.where(tensor == 0, 0, grad_tensor)
            grad = np.copy(grad_tensor)
            pre.append((name, (temp_t - (temp* grad))))
            p.grad.data = torch.from_numpy(grad_tensor).cuda()

        optimizer.step()

        for name, p in srcnn.named_parameters():
            if 'mask' in name:
                continue
            if 'label' in name:
                continue
            tensor = p.data.cpu().numpy()
            post.append((name, tensor))


        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))




def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = srcnn(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return (avg_psnr / len(testing_data_loader))


def checkpoint(trainCount, epoch):
    model_out_path = "saves/model_epoch_{}_{}.pth".format(trainCount, epoch)
    torch.save(srcnn, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

epoch_list_initial = []
psnr_list_initial = []
# initial Training
print("--- Initial training ---")
for epoch in range(1, opt.epochs):
    train(epoch)
    psnr = test()
    epoch_list_initial.append(epoch)
    psnr_list_initial.append(psnr)
    if(epoch%100==0):
        checkpoint(0, epoch)
util.log(opt.log, "initial_accuracy {}".format(psnr))

f = open("result_initial.csv", 'w')
print(epoch_list_initial)

for i in range(0, len(epoch_list_initial)):
    f.write(str(epoch_list_initial[i]) + ',' + str(psnr_list_initial[i]) + "\n")

f.close()

for i in range(5):
    # Pruning
    srcnn.prune_by_std(opt.sensitivity)
    psnr = test()
    util.log(opt.log, "accuracy_after_pruning {}".format(psnr))
    print("--- After pruning ---")
    util.print_nonzeros(srcnn)


    # Retrain
    print("--- Retraining ---")
    optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer

    epoch_list_retrain = []
    psnr_list_retrain = []

    for epoch in range(1, opt.epochs + 1):
        train(epoch)
        psnr = test()
        epoch_list_retrain.append(epoch)
        psnr_list_retrain.append(psnr)
        if(epoch%100==0):
            checkpoint(1, epoch)
    util.log(opt.log, "accuracy_after_retraining {}".format(psnr))

    print("--- After Retraining ---")
    util.print_nonzeros(srcnn)

torch.save(srcnn, "saves/pruned.pth")