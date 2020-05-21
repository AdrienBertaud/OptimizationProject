# Helper functions for training of ResNet and traditional CNN models
import torch
from torch import nn, optim
import time
from sklearn.metrics import f1_score
from imag_processing import *
from torch import nn, optim


# Training
def train_cnn(net, train_iter,test_iter, batch_size, optimizer, device, num_epochs):
    """Training functions for Lenet and AlexNet """
    net = net.to(device)
    net.train()
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_f1score = 0.0
    #test_f1score_plot = []
    #test_acc_plot = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, train_f1score_sum, n, batch_count, start = 0.0, 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            #train_l_new, train_acc_new, train_f1score_batch = 0.0, 0.0, 0.0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_new = l.cpu().item()
            train_l_sum += train_l_new
            train_acc_new = (y_hat.argmax(dim=1) == y).sum().cpu().item()/y.shape[0]
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            train_f1score_batch = f1_score(y.cpu(), y_hat.argmax(dim=1).long().cpu(), average='micro')
            train_f1score_sum += train_f1score_batch
            n += y.shape[0]
            batch_count += 1
            #print('loss new %.4f, train acc new %.4f, train f1 score new %.3f' % (train_l_new, train_acc_new, train_f1score_batch))
        test_acc, test_f1score = evaluate_accuracy(test_iter, net)
        # Save the best model parameter after every epoch
        if test_f1score > best_f1score:
            best_f1score = test_f1score
            #PATH = "./model_para/CNN_best.pt"
            #torch.save(net.state_dict(), PATH)
        #test_f1score_plot.append(test_f1score) #List for plot 
        #test_acc_plot.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, train f1 score %.3f, test acc %.3f, test f1score %.3f, best f1 score %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, train_f1score_sum / batch_count, test_acc, test_f1score, best_f1score, time.time() - start))
        




# Training
def train_resnet(net, train_iter,test_iter, batch_size, optimizer, device, num_epochs):
    """Training functions for ResNet """
    net = net.to(device)
    net.train()
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_f1score = 0.0
    #test_f1score_plot = []
    #test_acc_plot = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, train_f1score_sum, n, batch_count, start = 0.0, 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            #train_l_new, train_acc_new, train_f1score_batch = 0.0, 0.0, 0.0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_new = l.cpu().item()
            train_l_sum += train_l_new
            train_acc_new = (y_hat.argmax(dim=1) == y).sum().cpu().item()/y.shape[0]
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            train_f1score_batch = f1_score(y.cpu(), y_hat.argmax(dim=1).long().cpu(), average='micro')
            train_f1score_sum += train_f1score_batch
            n += y.shape[0]
            batch_count += 1
            #print('loss new %.4f, train acc new %.4f, train f1 score new %.3f' % (train_l_new, train_acc_new, train_f1score_batch))
        test_acc, test_f1score = evaluate_accuracy(test_iter, net)
        if test_f1score > best_f1score:
            best_f1score = test_f1score
            #PATH = "./model_para/ResNet_best.pt"
            #torch.save(net.state_dict(), PATH)
        #test_f1score_plot.append(test_f1score) #List for plot 
        #test_acc_plot.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, train f1 score %.3f, test acc %.3f, test f1score %.3f, best f1 score %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, train_f1score_sum / batch_count, test_acc, test_f1score, best_f1score, time.time() - start))




def evaluate_accuracy(data_iter, net, device=None):
    """Evaluate F1-score and accuracy on the validation set """
    if device is None and isinstance(net, nn.Module):
        # Using net's device
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    f1score_sum, n_batch = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # Evaluating mode
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                f1score_sum += f1_score(y.cpu(), net(X.to(device)).argmax(dim=1).long().cpu(), average='micro')
                #print(f1_score(y.cpu(), net(X.to(device)).argmax(dim=1).long().cpu(), average='micro'))
                net.train() # Return to training mode
            n += y.shape[0]
            n_batch += 1
    return acc_sum / n, f1score_sum / n_batch