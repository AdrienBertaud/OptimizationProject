import time
import torch
import src.utils
from importlib import reload
reload(src.utils)
from src.utils import eval_accuracy

def train(model, criterion, optimizer, optimizerName, dataloader, batch_size, n_epochs=1000, verbose=True, log_frequency=50):

    print("optimizer.type = ", optimizerName)

    with torch.set_grad_enabled(True):

        model.train()

        since = time.time()

        epoch_now = 0

        for epoch_now in range(n_epochs):

            n_batchs = len(dataloader)
            dataloader.idx = 0

            for i in range(n_batchs):

                if optimizerName == 'lbfgs':

                    def closure():
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        loss = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
                        return loss

                    optimizer.step(closure)
                    loss = compute_minibatch_gradient(model, criterion, dataloader, dataloader.batch_size)
                else:
                    optimizer.zero_grad()

                    loss = compute_minibatch_gradient(model, criterion, dataloader, batch_size)

                    optimizer.step()

                if loss < 1e-3:
                    print("loss very small, we stop learning at epoch ", epoch_now)
                    break;

            if epoch_now%log_frequency == 0 and verbose:
                now = time.time()
                loss_tot, acc_tot = eval_accuracy(model, criterion, dataloader)
                print('%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f'%(
                        epoch_now+1, n_epochs, now-since, loss_tot, acc_tot))
                since = time.time()

                if acc_tot == 100.:
                    print("accuracy of 100%, we stop learning at epoch ", epoch_now)
                    break;

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













