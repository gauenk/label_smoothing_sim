"""

A common set of shared pytorch functions
"""

# python imports
import importlib.util
import os,warnings
from functools import partial
import os.path as osp
import numpy as np
import numpy.random as npr

# torch imports
import torch as th
import torch.nn as nn
from torchvision import datasets,transforms

def pytorch_datasetSubset_to_numpy(subset):
    """
    Get the numpy data and labels from a torch.utils.data.dataset.Subset
    """
    batches = [elem for elem in subset]
    n = len(subset.dataset)
    d = subset.dataset[0][0].size()
    batch_size = batches[0][0].size()[0]
    data = np.zeros( (n,) + d, dtype=np.float )
    labels = np.zeros( n, dtype=np.int)
    prev = 0
    for batch in batches:
        start = prev
        end = prev + batch_size
        data[start:end,:] = batch[0]
        labels[start:end] = batch[1]
        prev = end
    return data,labels

def pytorch_model_outputs(model,data_loader,device):
    """
    run the model over the data
    """
    # test a classifier
    model.eval()
    cls,ftr,final = [],[],[]
    with th.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            raw_output = model(data)
            # unpack the batch
            for i in range(len(raw_output[0])):
                cls.append(raw_output[0][i])
                ftr.append(raw_output[1]['ftr'][i])
                final.append(raw_output[1]['final'][i])
    cls = np.array(cls)
    ftr = np.array(ftr)
    final = np.array(final)
    outputs = {'cls':cls,'ftr':ftr, 'final':final}
    return outputs

def save_pytorch_checkpoint(path,epoch,model_state_dict,optim_state_dict,loss,**kwargs):
    save_dict = {'epoch': epoch, 'model_state_dict': model_state_dict,
                 'optimizer_state_dict': optim_state_dict, 'loss': loss}
    save_dict += kwargs
    th.save(save_dict,path)

def load_pytorch_checkpoint(path,strict=True):    
    checkpoint = th.load(path)
    model.load_state_dict(checkpoint['model_state_dict'],strict=strict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model,optimizer,epoch,loss,checkpoint

def save_pytorch_model(model,path):
    pdir = osp.dirname(path)
    if not osp.exists(pdir):
        os.makedirs(pdir)
    th.save(model.state_dict(),path)

def load_pytorch_model(model_py,model_th):
    """
    Load the pytorch model from a (i) python file or (ii) saved th file.
    """
    model = None
    if model_py is None and model_th is None:
        raise ValueError("We need either the python file's path or torch snapshot path")
    if model_th is not None: # load snapshot if we can
        model = th.load(model_th)
    else:
        # load python object (the network) from file
        spec = importlib.util.spec_from_file_location("module.name", model_py)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        uninitializedModel = foo.get_network()
        model = uninitializedModel
    return model


def get_train_subset_indices(train_loader,n=10000):
    """
    Get indices for a representative subset of the classification training set
    """
    data = train_loader.dataset.data
    labels = train_loader.dataset.targets
    if type(data) is th.Tensor:
        data = data.detach().numpy()
        labels = labels.detach().numpy()
    if type(labels) is list:
        labels = np.array(labels)

    d = data[0].shape
    #print(data[0].shape)

    # number of samples per class
    prop_cls = np.bincount(labels)    
    #print(prop_cls,np.sum(prop_cls))
    prop_cls = prop_cls / np.sum(prop_cls)
    prop_cls = (prop_cls * n).astype(np.int)
    #print(prop_cls)

    # grab first %
    # sub_data = np.zeros( (n,) + d , dtype=np.float )
    # sub_labels = np.zeros( (n) , dtype=int )
    indices = np.zeros( (n) , dtype=int )
    prev = 0
    for idx,n_cls in enumerate(prop_cls):
        #print("idx,cls",idx,n_cls,np.where(labels == idx)[0])
        start = prev
        end = start + n_cls
        ncls_samples_indices = np.where(labels == idx)[0][:n_cls]
        indices[start:end] = ncls_samples_indices
        # sub_data[start:end,:] = data[ncls_samples_indices,:]
        # sub_labels[start:end] = labels[ncls_samples_indices]
        # print(sub_labels[start:end],len(sub_labels[start:end]))
        prev += n_cls
    return indices

class PytorchModelWrapper(nn.Module):
    """
    spoof additional functionality for pytorch models
    namely, let me access nn-modules by string-name,
    rather than expicitly iterating over a generator
    """

    def __init__(self,model):
        super(type(self),self).__init__()
        self.th_model = model

    def module_dict(self,module_name):
        for layer in self.th_model.named_modules():
            if layer[0] == module_name:
                return layer
        # print error message
        for layer in self.th_model.named_modules():
            print(layer[0])
        raise KeyError("No module named [{}] in the model.".format(module_name))

class FeatureExtractor():

    def __init__(self,in_model,feature_layers, to_numpy = True):
        self.th_model = in_model
        if type(in_model) is not PytorchModelWrapper:
            self.pyw_model = PytorchModelWrapper(in_model)
        else:
            self.th_model = in_model.th_model
            self.pyw_model = in_model
        self.th_model.eval()
        print(self.pyw_model.eval)
        self.activations = {}
        for layer_name in feature_layers:
            self.register_layer(layer_name)
        self.to_numpy = to_numpy

    def register_layer(self,l_name):
        layer = self.pyw_model.module_dict(l_name)
        l_hook = self._get_activation(layer[0])
        layer[1].register_forward_hook(l_hook)
        
    def _get_activation(self,name):
        def hook(module, inputs, output):
            self.activations[name] = output.data.cpu().detach()
        return hook

    def __call__(self,inputs):
        model_outputs = self.pyw_model.th_model(inputs)
        activations = self.activations
        akeys = list(activations.keys())
        for key in akeys:
            if self.to_numpy:
                activations[key] = activations[key]\
                                   .squeeze().detach().cpu().numpy()
            else:
                activations[key] = activations[key]\
                                   .squeeze().detach().cpu()
        if self.to_numpy:
            model_outputs = model_outputs.squeeze().detach().cpu().numpy()
                
        # flatten if we can
        if len(akeys) == 1:
            activations = activations[akeys[0]]
        return model_outputs,activations

    def eval(self):
        self.th_model.eval()

    def train(self):
        self.th_model.train()



#---------------------
# DATASET FUNCTIONS
#---------------------

def target_noise_transform(flipped_indices,flipped_labels,target,index):
    if index in flipped_indices:
        fl = flipped_labels[np.where(flipped_indices == index)][0]
        return fl
    else:
        return target

def apply_label_noise_level(cfg,ordered_targets):

    N = cfg.n_train_samples
    noise_level = cfg.label_noise

    # get number of flipped items and their index

    nflipped = int(N * noise_level)
    if nflipped == 0:
        return None
    flipped_indices = npr.permutation(N)[:nflipped]

    # create the new, flipped label which are saved for each program iteration
    flipped_labels = np.zeros(nflipped).astype(np.int)
    for idx in range(nflipped):
        ds_idx = flipped_indices[idx]
        target = ordered_targets[ds_idx]
        perm = npr.permutation(cfg.nclasses).astype(np.int)
        flipped_labels[idx] = perm[np.where(perm != target)][0]
    
    # TODO: cache the new labels
    warnings.warn("We should be caching our randomly generated labels by exp uuid.")

    # return the transfer functino 
    label_xfer = partial(target_noise_transform,flipped_indices,flipped_labels)
    return label_xfer

def get_features_from_raw(outputs,layer_name='feature_layer'):
    return outputs[layer_name]



