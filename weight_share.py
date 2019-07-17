import argparse
import os
import torch
import numpy as np
from math import log10
from torch.autograd import Variable
import torch.nn as nn
from srcnn_model import SRCNN
from quantization import apply_weight_sharing, all_dequantization, module_dequantization
import util
from torch.utils.data import DataLoader
from srcnn_data import get_training_set, get_test_set
import torch.optim as optim
from torchsummary import summary
import copy

parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
parser.add_argument('model', type=str, help='path to saved pruned model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--output', default='saves/model_after_weight_sharing_6bits.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
model = torch.load(args.model)
train_set = get_training_set(2)
test_set = get_test_set(2)
training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=50, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=10, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
initial_optimizer_state_dict = optimizer.state_dict()

def train(model, epoch):
    epoch_loss = 0
    index_model = copy.deepcopy(model)

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        all_dequantization(index_model)

        optimizer.zero_grad()
        model_out = index_model(input)
        loss = criterion(model_out, target)
        epoch_loss += loss.item()
        loss.backward()

        for (name, p), (new_name, new_p) in zip(index_model.named_parameters(), model.named_parameters()):
            centroid_grad = np.zeros(2**6)
            if 'mask' in name:
                continue
            if 'weight' in name:
                weight = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(weight == 0, 0, grad_tensor)
            if 'label' in name:
                label_dev = new_p.device
                label = p.data.cpu().numpy()
                for i in range(2 ** 6):
                    gradient = np.where(weight != label[i], 0, grad_tensor)
                    centroid_grad[i] = gradient.mean()
                    label[i] = label[i] - (centroid_grad[i] * 0.01)
                new_p.data = torch.from_numpy(np.array(label)).to(label_dev)

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        copy_model = copy.deepcopy(model)
        all_dequantization(copy_model)
        prediction = copy_model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return (avg_psnr / len(testing_data_loader))


def checkpoint(trainCount, epoch):
    model_out_path = "saves/model_epoch_{}_{}.pth".format(trainCount, epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

# Define the model
print('accuracy before weight sharing')
#util.test(model, use_cuda)

#print("model summary")
#summary(model, (3, 256, 256))

# Weight sharing
apply_weight_sharing(model)
print('accuacy after weight sharing')
util.test(model, use_cuda)


# Save the new model
os.makedirs('saves', exist_ok=True)
torch.save(model, args.output)


avg_psnr = 0
for batch in testing_data_loader:
    input, target = Variable(batch[0]), Variable(batch[1])
    if use_cuda:
        input = input.cuda()
        target = target.cuda()

    prediction = model(input)
    mse = criterion(prediction, target)
    psnr = 10 * log10(1 / mse.item())
    avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

epoch_list_retrain = []
psnr_list_retrain = []

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer

epoch_list_retrain = []
psnr_list_retrain = []

for epoch in range(1, 100):
    train(model, epoch)
    psnr = test()
    epoch_list_retrain.append(epoch)
    psnr_list_retrain.append(psnr)
    if(epoch%100==0):
        checkpoint(1, epoch)


# Weight sharing
apply_weight_sharing(model)

print('accuacy after weight sharing')
util.test(model, use_cuda)

print("--- After Retraining ---")
util.test(model, use_cuda)

avg_psnr = 0
for batch in testing_data_loader:
    input, target = Variable(batch[0]), Variable(batch[1])
    if use_cuda:
        input = input.cuda()
        target = target.cuda()

    prediction = model(input)
    mse = criterion(prediction, target)
    psnr = 10 * log10(1 / mse.item())
    avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
#summary(model, (3, 256, 256))


# Save the new model
os.makedirs('saves', exist_ok=True)
torch.save(model, args.output)