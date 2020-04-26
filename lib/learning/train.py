"""
Demo for Miami, FL 02/2020

Contains API for training a Pytorch deep learning model
"""

# torch imports
import torch as th
import torch.nn.functional as F

def thtrain_cls(cfg, model, device, train_loader, optimizer, epoch):
    # train a classifier
    model.train()
    idx = 0
    print("N samples:", len(train_loader.dataset.data))
    for batch_idx, (data, target) in enumerate(train_loader):
        idx += cfg.batch_size
        data, target = data.to(device), target.to(device)
        target = target.long()#.squeeze_()
        optimizer.zero_grad()
        output = model(data)
        # print_output_target(output,target)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % cfg.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def print_output_target(output,target):
    # print(output.shape,"output")
    # print(target.shape,"target")
    g = th.argmax(output,1)
    print(g,target)
    print(g == target)
