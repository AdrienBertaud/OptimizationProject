import time
import torch
import src.utils
from importlib import reload
reload(src.utils)
from src.utils import accuracy

def train(model, criterion, optimizer, optimizerName, dataloader, batch_size, n_epochs=1000, verbose=True, log_frequency=50):

    print("optimizer.type = ", optimizerName)

    with torch.set_grad_enabled(True):

        model.train()

        since = time.time()

        iter_now = 0

        for iter_now in range(n_epochs):

            n_batchs = len(dataloader)
            dataloader.idx = 0
            acc_tot = .0
            loss_tot = .0

            for i in range(n_batchs):

                if optimizerName == 'lbfgs':

                    def closure():
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        loss,acc = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
                        return loss

                    optimizer.step(closure)
                    loss,acc = compute_minibatch_gradient(model, criterion, dataloader, dataloader.batch_size)
                else:
                    optimizer.zero_grad()

                    loss,acc = compute_minibatch_gradient(model, criterion, dataloader, batch_size)

                    optimizer.step()

                acc_tot += acc
                loss_tot += loss

            acc_tot = acc_tot/n_batchs
            loss_tot = loss_tot/n_batchs

            if iter_now%log_frequency == 0 and verbose:
                now = time.time()
                print('%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f'%(
                        iter_now+1, n_epochs, now-since, loss_tot, acc_tot))
                since = time.time()

            if acc_tot == 100.:
                print("accuracy of 100%, we stop learning at epoch ", n_epochs)
                break;

    return iter_now

def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss,acc = 0,0

    #inputs, targets = inputs.cuda(), targets.cuda()
    inputs,targets = next(dataloader)

    targets_indices = torch.argmax(targets,1)

    y_hat = model(inputs)

    E = criterion(y_hat,targets_indices)
    E.backward(retain_graph=True)

    loss = E.item()
    acc = accuracy(y_hat,targets_indices)

    return loss, acc













