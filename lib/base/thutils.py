"""

A common set of shared pytorch functions
"""

# python imports
import importlib.util
import os,warnings
import os.path as osp
import numpy as np

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
    cls,fc1,fc2 = [],[],[]
    with th.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            raw_output = model(data)
            # unpack the batch
            for i in range(len(raw_output[0])):
                cls.append(raw_output[0][i])
                fc1.append(raw_output[1]['fc1'][i])
                fc2.append(raw_output[1]['fc2'][i])
    cls = np.array(cls)
    fc1 = np.array(fc1)
    fc2 = np.array(fc2)
    outputs = {'cls':cls,'fc1':fc1, 'fc2':fc2}
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
    data = train_loader.dataset.train_data.detach().numpy()
    labels = train_loader.dataset.train_labels.detach().numpy()
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

#
# MNIST
#

def target_noise_transform(info,target):
    pass

def apply_label_noise_level(cfg):
    warnings.warn(f"Noise Level {cfg.noise_level}: Hardcoded N = 60k")
    N = 60000
    noise_level = cfg.noise_level
    nflipped = N * noise_level
    np.permutation(

def load_mnist_data(cfg):
    
    apply_label_noise = None
    if cfg.label_noise is not None:
        apply_label_noise = apply_label_noise_level(cfg)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    train_loader = th.utils.data.DataLoader(
        datasets.MNIST(f'{cfg.root_path}/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),
                       target_transform=apply_label_noise),
        batch_size=cfg.batch_size, shuffle=True, **kwargs)
    train_subset_indices = get_train_subset_indices(train_loader)
    train_subset_loader = th.utils.data.DataLoader(
        th.utils.data.Subset(
            datasets.MNIST(f'{cfg.root_path}/data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
                           target_transform=apply_label_noise),
            train_subset_indices
            ),
        batch_size=cfg.batch_size, shuffle=False, **kwargs)
    test_loader = th.utils.data.DataLoader(
        datasets.MNIST(f'{cfg.root_path}/data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=cfg.test_batch_size, shuffle=False, **kwargs)
    return train_loader,test_loader,train_subset_loader

def get_features_from_raw(outputs,layer_name='fc1'):
    return outputs[layer_name]



