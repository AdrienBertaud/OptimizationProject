import time
import torch
import utils.utils
from importlib import reload
reload(utils.utils)
from utils.utils import eval_accuracy

def train(model, criterion, optimizer, optimizerName, dataloader, batch_size, n_epochs=1000, verbose=True):

    eval_frequency=50
    loss_condition = 1e-3

    print("optimizer.type = ", optimizerName)

    with torch.set_grad_enabled(True):

        model.train()

        if verbose:
            since = time.time()

        epoch_now = 0

        for epoch_now in range(n_epochs):

            n_batchs = len(dataloader)
            dataloader.idx = 0
            total_loss = .0

            for i in range(n_batchs):

                if optimizerName == 'lbfgs':

                    def closure():
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        loss = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
                        return loss

                    optimizer.step(closure)
                    compute_minibatch_gradient(model, criterion, dataloader, dataloader.batch_size)
                else:
                    optimizer.zero_grad()

                    compute_minibatch_gradient(model, criterion, dataloader, batch_size)

                    optimizer.step()


            if epoch_now%eval_frequency == 0:

                loss_train, acc_train = eval_accuracy(model, criterion, dataloader)

                if verbose:
                    now = time.time()
                    print('%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f'%(
                            epoch_now+1, n_epochs, now-since, loss_train, acc_train))

                if loss_train <= loss_condition:
                    print("loss is egal inferior to {0}, we stop learning at epoch {1}", loss_condition, epoch_now)
                    break;

                since = time.time()

    return epoch_now

def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss = .0

    inputs,targets = next(dataloader)

    targets_indices = torch.argmax(targets,1)

    y_hat = model(inputs)

    E = criterion(y_hat,targets_indices)
    E.backward(retain_graph=True)

    loss = E.item()

    return loss













