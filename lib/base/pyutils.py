"""

A common set of shared pytorch functions
"""

# python imports
import importlib.util

# torch imports
import torch as th
import torch.nn as nn

def pytorch_model_outputs(model,data_loader):
    """
    run the model over the data
    """
    # test a classifier
    model.eval()
    idx = 0
    outputs = []
    with th.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            outputs.append(output)
            idx += 1
    return outputs

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


class PytorchModelWrapper(nn.Module):
    """
    spoof additional functionality for pytorch models
    namely, let me access nn-modules by string-name,
    rather than expicitly iterating over a generator
    """

    def __init__(self,model):
        super(type(self),self).__init__()
        self.model = model

    def module_dict(self,module_name):
        for layer in self.model.named_modules():
            if layer[0] == module_name:
                return layer
        # print error message
        for layer in self.model.named_modules():
            print(layer[0])
        raise KeyError("No module named [{}] in the model.".format(module_name))

class FeatureExtractor():

    def __init__(self,model,feature_layers):
        self.py_model = model
        if type(model) is not PytorchModelWrapper:
            self.pyw_model = PytorchModelWrapper(model)
        self.pyw_model.eval()
        self.activations = {}
        for layer_name in feature_layers:
            self.register_layer(layer_name)

    def register_layer(self,l_name):
        layer = self.pyw_model.module_dict(l_name)
        l_hook = self._get_activation(layer)
        layer[1].register_forward_hook(l_hook)
        
    def _get_activation(self,name):
        def hook(model, inputs, output):
            self.activations[name] = output.cpu().detach()
        return hook

    def __call__(self,inputs):
        model_outputs = self.pyw_model.model(inputs)
        activations = self.activations
        akeys = list(activations.keys())
        if len(akeys) == 1:
            activations = activations[akeys[0]].squeeze().detach().cpu().numpy()
        return model_outputs,activations
