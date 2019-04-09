'''Train CIFAR10 with PyTorch.'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle 

from models import *
from utils import progress_bar

def _build_network(model_name):
    net = None 
    if model_name == 'VGG16':
        net = VGG('VGG16')
    elif model_name == 'ResNet18':
        net = ResNet18()
    elif model_name == 'ResNet50':
        net = ResNet50() 
    elif model_name == 'ResNet101':
        net = ResNet101()
    elif model_name == 'MobileNetV2':
        net = MobileNetV2()
    elif model_name == 'GoogLeNet':
        net = GoogLeNet()

    assert net != None
    return net 

# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('model', help='the name of model to be trained')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', 
        help='resume from checkpoint')
parser.add_argument('--checkpoint_dir', default=None, help='the directory of'
        ' checkpoint models')
parser.add_argument('--data_dir', default=None, help='the directory storing data')
parser.add_argument('--num_epoches', default=10, type=int, help='the number of'
        ' epoches for training')
parser.add_argument('--batchsize', default=128, type=int, help='the batch size of'
        ' each training iteration')

args = parser.parse_args()

work_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(work_dir)

if args.checkpoint_dir is None:
    checkpoint_dir = os.path.join(work_dir, "checkpoint", args.model)
else:
    checkpoint_dir = args.checkpoint_dir 

if args.data_dir is None:
    data_dir = os.path.join(work_dir, "data")
else:
    data_dir = args.data_dir 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, 
        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, 
        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
        'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = _build_network(args.model)
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        num_batches += 1

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # print('Loss: ', train_loss / num_batches)
    return (train_loss / num_batches)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'ckpt.t7'))
        best_acc = acc

    return acc 

training_curve = []
for epoch in range(start_epoch, start_epoch + args.num_epoches):
    epoch_loss = train(epoch)
    epoch_acc = test(epoch)
    training_curve.append({
        'epoch': epoch,
        'loss': epoch_loss,
        'acc': epoch_acc
    })

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir) 
with open(os.path.join(checkpoint_dir, 
    'loss_acc_{}_{}'.format(start_epoch, start_epoch + args.num_epoches)), 'wb') as f:
    pickle.dump(training_curve, f) 

