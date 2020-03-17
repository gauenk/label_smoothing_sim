"""
Demo for Miami, FL 02/2020

Contains API for training a Pytorch deep learning model
"""

# torch imports
import torch as th
import torch.nn.functional as F

def pytest_cls(args, model, device, test_loader):
    # test a classifier
    model.eval()
    test_loss = 0
    correct = 0
    with th.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.long()#.squeeze_()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss,correct,len(test_loader.dataset)

