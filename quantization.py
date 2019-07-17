import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
from torch.nn.modules.module import Module

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    return dot(v, v)

def squared_distance(input, value):
    return sum_of_squares(vector_subtract(input, value))

def weight_sharing(input, space):
    def classify(self, input):
        return min(range(self.k), key = lambda i: squared_distance(input, space[i]))


def apply_weight_sharing(model, bits=6):
    """
    Applies weight sharing to the given model
    """
    for module in model.children():
        #print("module", module)
        for name, p in module.named_parameters():
            #print("name : ", name)
            if 'weight' in name:
                weight_dev = module.weight.device
                old_weight = module.weight.data.cpu().numpy()
                reshape_weight = old_weight.reshape(1, old_weight.shape[0]*old_weight.shape[1]*old_weight.shape[2]*old_weight.shape[3])
                shape = reshape_weight.shape
                mat = csr_matrix(reshape_weight) if shape[0] < shape[1] else csc_matrix(reshape_weight)
                min_ = min(mat.data)
                #print("min", min_)
                max_ = max(mat.data)
                #print("max", max_)
                space = np.linspace(min_, max_, num=2**bits)
                #print("space")
                #print(space)
                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full", max_iter=1)
                label = (kmeans.fit(mat.data.reshape(-1,1))).cluster_centers_
                #print("label")
                #print(label)
                new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                mat.data = new_weight
                quantized_weight = mat.todense()
                for i in range(2**bits):
                    #print("i", i)
                    quantized_weight = np.where(quantized_weight == label[i], i+1, quantized_weight)
                #new_mat = csc_matrix(quantized_weight)
                #print("new_mat.data")
                #print(new_mat)
                #print("quantized_weight.shape")
                #print(quantized_weight.shape)
                #print("quantized_weight")
                #print(quantized_weight)
                module.weight.data = torch.from_numpy(
                    np.array(quantized_weight).reshape(old_weight.shape[0], old_weight.shape[1], old_weight.shape[2],
                                                       old_weight.shape[3])).to(weight_dev)

            if 'label' in name:
                label_dev = module.label.device
                #print("---------------------------------------------------")
                #print("module label")
                #print(label)
                module.label.data = torch.from_numpy(np.array(label)).to(label_dev)


def all_dequantization(model):
    for module in model.children():
        for name, p in module.named_parameters():
            #print("name : ", name)
            if 'weight' in name:
                weight_dev = module.weight.device
                weight = module.weight.data.cpu().numpy()

            if 'label' in name:
                label_dev = module.label.device
                label = module.label.data.cpu().numpy()

        for i in range(2 ** 6):
            weight = np.where(weight == i+1, label[i], weight)

        for name, p in module.named_parameters():
            #print("name : ", name)
            if 'weight' in name:
                module.weight.data = torch.from_numpy(np.array(weight)).to(weight_dev)

            if 'label' in name:
                module.label.data = torch.from_numpy(np.array(label)).to(label_dev)



def module_dequantization(module):
    for name, p in module.named_parameters():
        #print("name : ", name)
        if 'weight' in name:
            weight_dev = module.weight.device
            weight = module.weight.data.cpu().numpy()

        if 'label' in name:
            label_dev = module.label.device
            label = module.label.data.cpu().numpy()

    for i in range(2 ** 6):
        weight = np.where(weight == i+1, label[i], weight)

    for name, p in module.named_parameters():
        #print("name : ", name)
        if 'weight' in name:
            module.weight.data = torch.from_numpy(np.array(weight)).to(weight_dev)

        if 'label' in name:
            module.label.data = torch.from_numpy(np.array(label)).to(label_dev)

