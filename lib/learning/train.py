"""
Demo for Miami, FL 02/2020

Contains API for training a Pytorch deep learning model
"""

# torch imports
import torch.nn.functional as F

def pytrain_cls(args, model, device, train_loader, optimizer, epoch):
    # train a classifier
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.long()#.squeeze_()
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape,"output")
        # print(target.shape,"target")
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        

