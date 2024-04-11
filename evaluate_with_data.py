import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim import lr_scheduler
import datetime
from datetime import datetime
import numpy as np
import copy
from scipy import stats
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchvision import datasets, transforms, models
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url




parser = argparse.ArgumentParser(description='Evaluate feature metrics on pre-trained torchvision architectures')

parser.add_argument('--dataset', default="CIFAR100",
                    help='which torchvision dataset to use')
parser.add_argument('--arch', default="resnet50",
                    help='which torchvision pretrained architecture to use')
parser.add_argument('--finetune', type=int, default=0,
                    help='finetune feature extractor with small lr')
parser.add_argument('--train-batch-size', type=int, default=8,
                    help='training batch-size to use')
parser.add_argument('--eval-batch-size', type=int, default=8,
                    help='evaluation batch-size to use')
parser.add_argument('--accumulation-steps', type=int, default=8,
                    help='accumulate gradients this many times before updating parameters')
parser.add_argument('--lr-classifier', type=float, default=0.001,
                    help='learning rate for new classifier head')
parser.add_argument('--lr-extractor', type=float, default=0.0001,
                    help='learning rate for finetuning feature extractor')
parser.add_argument('--epochs-max', type=int, default=15,
                    help='max epochs for training classifier or finetuning')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight-decay for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for training')


parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--verbose', default=0, type=int,
                    help="whether to print to std out")



args = parser.parse_args()


kwargsUser = {}
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string

with open('commandline_args.txt', 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

#use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
torch.cuda.empty_cache()
print ("cuda: ", use_cuda)

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
        m.training=False

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()

def compute_metrics(cur_model, compact_dataloaders):
    intra_class_similarities = {}
    intra_class_vectors = {}
    batch_prototypes = {}
    cur_model.multi_out = 1

    with torch.no_grad():
        for k, cdl in compact_dataloaders.items():
            if k % 10 == 0:
                print (k)
            for z, data in enumerate(cdl, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                num_samples = len(inputs)

                #careful of normalization

                feat_vec, _ = cur_model(inputs)  ## get feature vectors from the model

                feat_vec_unit = F.normalize(feat_vec, dim=1)
                #all_pairs = torch.matmul(feat_vec_unit, feat_vec_unit.t())
                #all_pairs_nondiag = all_pairs.cpu().masked_select(~torch.eye(num_samples, dtype=bool)).view(num_samples,num_samples-1).flatten()

                if k not in intra_class_similarities:
                    intra_class_similarities[k] = []
                    intra_class_vectors[k] = []

                if k not in batch_prototypes:
                    batch_prototypes[k] = []

                #intra_class_similarities[k].extend(all_pairs_nondiag.cpu().numpy().tolist())
                intra_class_vectors[k].append(feat_vec_unit.cpu())
                batch_prototypes[k].append(torch.mean(feat_vec,dim=0).cpu())

            all_units_k = torch.cat(intra_class_vectors[k],dim=0).to(device)
            all_pairs_k = torch.matmul(all_units_k, all_units_k.t())
            all_pairs_nondiag_k = all_pairs_k.cpu().masked_select(~torch.eye(all_pairs_k.shape[0], dtype=bool)).view(all_pairs_k.shape[0],all_pairs_k.shape[0]-1).flatten()
            intra_class_similarities[k].extend(all_pairs_nondiag_k.cpu().numpy().tolist())


        inter_class_similarities = {}

        for k1, v1 in batch_prototypes.items():
            if k1 % 10 == 0:
                print (k1)
            for k2, v2 in batch_prototypes.items():
                if k1 != k2:
                    v1_ = torch.stack(v1).to(device)
                    v2_ = torch.stack(v2).to(device)
                    #print (v2_.shape)
                    v1_norm = F.normalize(v1_, dim=1)
                    v2_norm = F.normalize(v2_, dim=1)
                    all_pairs = torch.matmul(v1_norm, v2_norm.t()).flatten()
                    #print (all_pairs.shape)

                    if k1 not in inter_class_similarities:
                        inter_class_similarities[k1] = []

                    if k2 not in inter_class_similarities:
                        inter_class_similarities[k2] = []

                    inter_class_similarities[k1].extend(all_pairs.cpu().numpy().tolist())
                    inter_class_similarities[k2].extend(all_pairs.cpu().numpy().tolist())

    class_intra_similarity_means, class_inter_similarity_means = [], []
    class_intra_similarity_std, class_inter_similarity_std = [], []

    for class_id in intra_class_similarities:
        #print (intra_class_similarities[class_id][0])
        class_intra_similarity_means.append(np.mean(intra_class_similarities[class_id]))
        class_inter_similarity_means.append(np.mean(inter_class_similarities[class_id]))
        class_intra_similarity_std.append(np.std(intra_class_similarities[class_id]))
        class_inter_similarity_std.append(np.std(inter_class_similarities[class_id]))

    return np.mean(class_intra_similarity_means), np.mean(class_intra_similarity_std), np.mean(class_inter_similarity_means), np.mean(class_inter_similarity_std)

   


def reset_learnable(m, ft=0):
    for param in m.extractor.parameters():
        param.requires_grad = False
        m.apply(set_bn_eval)

    if ft:
        for param in m.extractor.parameters():
            param.requires_grad = True
        m.apply(set_bn_train)

    for param in m.linear.parameters():
        param.requires_grad = True



class TransferWrapper(nn.Module):
    def __init__(self, extractor, nftr, nclass):
        super(TransferWrapper, self).__init__()

        self.extractor = extractor
        self.nclass = nclass
        self.nftr = nftr
        self.multi_out = 0

        self.do = nn.Dropout(0.2)
        self.linear = nn.Linear(self.nftr, self.nclass)

    def forward(self, x):

        p = self.extractor(x)

        out = self.linear(self.do(p))

        if (self.multi_out):
            return p, out
        else:
            return out

def evaluate(model, loader):
    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += F.cross_entropy(outputs, labels).item()

    accuracy = 100 * correct / total
    loss /= len(loader)

    print('Accuracy: %.3f %%' % (accuracy))
    print('Loss: %.5f' % (loss))

    return accuracy, loss



def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #299 for inception
    #224 resnet

    if args.arch == "inception_v3":
        final_size = 299
    else:
        final_size = 224
    
    train_transform = transforms.Compose(
                [transforms.RandomResizedCrop(final_size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)
                 ])



    gen_transform = transforms.Compose(
                [transforms.Resize(final_size+32),
                 transforms.CenterCrop(final_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)
                 ])

    if (args.dataset == "CIFAR100"):

        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform)

        nclass=100
        nchannels = 3
        H, W = 32, 32
        targs_ds = trainset.targets

    elif (args.dataset == "CIFAR10"):
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=gen_transform)

        nclass=10
        nchannels = 3
        H, W = 32, 32
        targs_ds = trainset.targets

    elif (args.dataset == "STL10"):
        trainset = torchvision.datasets.STL10(root='../data', split='train', download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.STL10(root='../data', split='train', download=True, transform=train_transform)
        testset = torchvision.datasets.STL10(root='../data', split='test', download=True, transform=gen_transform)

        nclass=10
        nchannels = 3
        H, W = 96, 96
        targs_ds = trainset.labels

    elif (args.dataset == "FLOWERS"):
        trainset = torchvision.datasets.Flowers102(root='../data', split='train', download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.Flowers102(root='../data', split='train', download=True, transform=train_transform)
        valset = torchvision.datasets.Flowers102(root='../data', split='val', download=True, transform=gen_transform)
        testset = torchvision.datasets.Flowers102(root='../data', split='test', download=True, transform=gen_transform)

        trainset._labels.extend(valset._labels)
        trainset._image_files.extend(valset._image_files)
        eval_trainset._labels.extend(valset._labels)
        eval_trainset._image_files.extend(valset._image_files)

        nclass = 102
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset._labels

    elif (args.dataset == "OXFORD"):
        trainset = torchvision.datasets.OxfordIIITPet(root='../data', split='trainval', download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.OxfordIIITPet(root='../data', split='trainval', download=True, transform=train_transform)
        testset = torchvision.datasets.OxfordIIITPet(root='../data', split='test', download=True, transform=gen_transform)
        nclass=37
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset._labels
    elif (args.dataset == "NATURE"):
        trainset = torchvision.datasets.INaturalist(root='../data', version='2021_train_mini', download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.INaturalist(root='../data', version='2021_train_mini', download=True, transform=train_transform)
        testset = torchvision.datasets.INaturalist(root='../data', version='2021_valid', download=True, transform=gen_transform)
        nclass=1000
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset.all_categories

    elif (args.dataset == "CARS"):
        trainset = torchvision.datasets.StanfordCars(root='../data', split='train', download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.StanfordCars(root='../data', split='train', download=True, transform=train_transform)
        testset = torchvision.datasets.StanfordCars(root='../data', split='test', download=True, transform=gen_transform)
        nclass=196
        nchannels=3
        H,W = 224, 224
        targs_ds = trainset._samples
    elif (args.dataset == "CALTECH256"):
        trainset = torchvision.datasets.Caltech256(root='../data', split='train', download=True, transform=train_transform)
        eval_trainset = torchvision.datasets.Caltech256(root='../data', split='train', download=True, transform=train_transform)
        testset = torchvision.datasets.Caltech256(root='../data', split='test', download=True, transform=gen_transform)

        nclass=256
        nchannels = 3
        H, W = 96, 96
        targs_ds = trainset.labels

    elif (args.dataset == "EYEPACS"):
        trainset = torchvision.datasets.EyePacs(root='../data', split='train', download=True, transform=train_transform)
    else:
        raise ValueError("Unknown dataset")

    #split the trainset by label into separate subsets and create a separate dataloader for each class
    # Get unique labels
    if args.dataset != "CARS":
        unique_labels = set(targs_ds)
    else:
        unique_labels = set([c for (name, c) in targs_ds])
    # Create a dictionary to store dataloaders for each class
    compact_dataloaders = {}
    # Iterate over unique labels
    for label in unique_labels:
        # Get indices of data points with the current label
        indices = [i for i, target in enumerate(targs_ds) if target == label]
        # Create a subset of the trainset with the current label
        subset = Subset(trainset, indices)
        # Create a dataloader for the subset
        dataloader = DataLoader(subset, batch_size=args.eval_batch_size, shuffle=True)
        # Add the dataloader to the dictionary
        compact_dataloaders[label] = dataloader
        
    
    if args.arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model_ft = models.resnet18(weights=weights)
    elif args.arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model_ft = models.resnet50(weights=weights)
    elif args.arch =="inception_v3":
        weights = models.Inception_V3_Weights.DEFAULT
        model_ft = models.inception_v3(weights=weights)
        model_ft.aux_logits = False
    elif args.arch == "shufflenet_v2_x2_0":
        weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT
        model_ft = models.shufflenet_v2_x2_0(weights=weights)
    elif args.arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model_ft = models.convnext_tiny(weights=weights)
    elif args.arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model_ft = models.densenet121(weights=weights)
    elif args.arch == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT
        model_ft = models.swin_t(weights=weights)
    elif args.arch == "max_vit":
        weights = models.MaxVit_T_Weights.DEFAULT
        model_ft = models.maxvit_t(weights=weights)
    elif args.arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model_ft = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError("Unknown architecture")

    #print (model_ft)
    if args.arch not in ["swin_t", "max_vit", "densenet121"]:
        num_ftrs = model_ft.fc.in_features
        out_ftrs = nclass
        model_ft.fc = nn.Sequential()
    else:
        out_ftrs = nclass
        if args.arch == "swin_t":
            num_ftrs = model_ft.head.in_features
            model_ft.head = nn.Sequential()
        elif args.arch == "densenet121":
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Sequential()
        else:
            num_ftrs = 512
            model_ft.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                                                nn.Flatten(start_dim=1, end_dim=-1),
                                                nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True))


    #wrap feature extractor with new classification head.  Allows explicit return of feature vectors.
    cur_model = TransferWrapper(model_ft, num_ftrs, out_ftrs).to(device)

    if not args.finetune:
        print ("not finetuning")
        optimizer = optim.SGD(cur_model.linear.parameters(), lr=args.lr_classifier, momentum=args.momentum, weight_decay=args.weight_decay)
        #optimizer = optim.Adam(cur_model.parameters(), lr=0.00002)
    else:
        optimizer = optim.SGD([{'params': cur_model.extractor.parameters()},
                          {'params': cur_model.linear.parameters(), 'lr':args.lr_classifier}], lr=args.lr_extractor, momentum=args.momentum, weight_decay=args.weight_decay)
        #optimizer = optim.Adam(cur_model.parameters(), lr=0.00002)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # define a trainloader and testloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
    eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.eval_batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False)

    # write a generic training loop for me using cross entropy loss but also accounting for multi_out variable in transferwrapper which can output the feature vector layer as well
    cur_model.eval()


    cur_model.multi_out = 1
    compact_mean, compact_std, sep_mean, sep_std = compute_metrics(cur_model, compact_dataloaders)
    cur_model.multi_out = 0
    cur_model.train()
    reset_learnable(cur_model, ft=args.finetune)


    num_epochs = args.epochs_max
    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []
    acc_steps = args.accumulation_steps

    cur_model.zero_grad() 

    for epoch in range(1,num_epochs+1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if cur_model.multi_out:
                p, outputs = cur_model(inputs)
                loss = F.cross_entropy(outputs, labels)
            else:
                outputs = cur_model(inputs)
                loss = F.cross_entropy(outputs, labels)

            loss = loss / acc_steps
            loss.backward()
            if (i+1) % acc_steps == 0:             
                optimizer.step()                            
                cur_model.zero_grad()                       


            #if i < steps_per_epoch:
            #    scheduler.step()

            running_loss += loss.item()

            if i % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(labels), len(trainloader.dataset),
                       100. * i / len(trainloader), loss.item()))

        scheduler.step()
        train_losses.append(running_loss/len(trainloader))

        #print(f'Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss/len(trainloader):.3f}')
        #accuracy_train, loss_train = evaluate(cur_model, eval_trainloader)
        cur_model.eval()
        cur_model.multi_out = 0
        acc_train, loss_train = evaluate(cur_model, eval_trainloader)
        acc_test, loss_test = evaluate(cur_model, trainloader)
        
        train_accs.append(acc_train)
        train_losses.append(loss_train)
        test_accs.append(acc_test)
        test_losses.append(loss_test)

        if epoch in [5,10,15,25] and args.finetune:
            cur_model.multi_out = 1
            compact_mean, compact_std, sep_mean, sep_std = compute_metrics(cur_model, compact_dataloaders)

            #print(f"Overall Statistics:\nIntra-class Similarity: Mean = {np.mean(class_)}, Std = {std_intra_sim}, Var = {var_intra_sim}\nInter-class similarity: Mean = {mean_inter_sim}, Std = {std_inter_sim}, Var = {var_inter_sim}\n")
            print(f"Intra-class Similarity: Mean = {compact_mean}, Std = {compact_std}")
            print(f"Inter-class Similarity: Mean = {sep_mean}, Std = {sep_std}\n")

        cur_model.multi_out = 0
        cur_model.train()
        reset_learnable(cur_model, ft=args.finetune)


        #if finetuning, metrics are updated, otherwise, metric evaluation on pre-trained extractor is kept and repeated in text file
        with open('similarity_statistics_{}_{}.txt'.format(args.arch,args.dataset), 'a') as f:
            if os.stat('similarity_statistics_{}_{}.txt'.format(args.arch,args.dataset)).st_size == 0:
                f.write('Dataset, Architecture, Train Batch Size, Classifier LR, Finetune, Feature LR, MaxEpochs, Weight Decay, Momentum, Eval Batch Size, Epoch, Training Acc, Test Acc, Intra-class Similarity Mean, Intra-class Similarity Std, Inter-class Similarity Mean, Inter-class Similarity Std\n')
            f.write(f"{args.dataset}, {args.arch}, {args.train_batch_size}, {args.lr_classifier}, {args.finetune}, {args.lr_extractor}, {args.epochs_max}, {args.weight_decay}, {args.momentum}, {args.eval_batch_size}, {epoch}, {train_accs[-1]:.5f}, {test_accs[-1]:.5f}, {compact_mean:.6f}, {compact_std:.6f}, {sep_mean:.6f}, {sep_std:.6f}\n")
            
if __name__ == '__main__':
    main()

