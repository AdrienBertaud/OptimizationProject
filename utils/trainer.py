import time
import torch
import utils.accuracy

from importlib import reload
reload(utils.accuracy)

from utils.accuracy import eval_accuracy

def train(model, loss_function, optimizer, optimizer_name, data_loader, batch_size):

    begin = time.time()

    n_batchs = len(data_loader)
    max_epochs=10000//n_batchs
    eval_frequency=400//n_batchs
    loss_condition = 1e-3

    with torch.set_grad_enabled(True):

        model.train()

        since = time.time()

        epoch_now = 0

        for epoch_now in range(max_epochs):

            data_loader.idx = 0

            for i in range(n_batchs):

                if optimizer_name == 'lbfgs':

                    def closure():
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        loss = compute_minibatch_gradient(model, loss_function, data_loader, batch_size)
                        return loss

                    optimizer.step(closure)
                    compute_minibatch_gradient(model, loss_function, data_loader, data_loader.batch_size)
                else:
                    optimizer.zero_grad()

                    compute_minibatch_gradient(model, loss_function, data_loader, batch_size)

                    optimizer.step()

            if epoch_now%eval_frequency == 0:

                loss_train, acc_train = eval_accuracy(model, loss_function, data_loader)

                now = time.time()
                print('%d/%d, took %.0f/%.0f seconds, train_loss: %.1e, train_acc: %.2f'%(epoch_now+1, max_epochs, now-since, now-begin, loss_train, acc_train))

                if loss_train <= loss_condition:
                    print('loss is egal or inferior to %d, we stop learning at epoch %d'%(loss_condition, epoch_now))
                    break;

                since = time.time()

    return epoch_now, time.time()-begin

def compute_minibatch_gradient(model, loss_function, data_loader, batch_size):
    loss = .0

    inputs,targets = next(data_loader)

    targets_indices = torch.argmax(targets,1)

    y_hat = model(inputs)

    E = loss_function(y_hat,targets_indices)
    E.backward(retain_graph=True)

    loss = E.item()

    return loss













