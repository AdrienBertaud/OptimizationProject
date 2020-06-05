import time
import torch

def train(model, criterion, optimizer, optimizerName, dataloader, batch_size, n_iters=1000, verbose=True, logFrequency=200):

    print("optimizer.type = ", optimizerName)

    model.train()
    acc_avg, loss_avg = 0, 0

    since = time.time()

    iter_now = 0

    for iter_now in range(n_iters):

        if optimizerName == 'lbfgs':

            batch_size_used = 0

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

        acc_avg = 0.9 * acc_avg + 0.1 * acc if acc_avg > 0 else acc
        loss_avg = 0.9 * loss_avg + 0.1 * loss if loss_avg > 0 else loss

        if iter_now%logFrequency == 0 and verbose:
            now = time.time()
            print('%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f'%(
                    iter_now+1, n_iters, now-since, loss_avg, acc_avg))
            since = time.time()

        if acc == 100.:
            print("accuracy of 100, we stop learning")
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

def accuracy(y_hat, targets):

    y_hat_indices = torch.argmax(y_hat,1)
    acc = (targets==y_hat_indices).float()

    return torch.mean(acc)*100.0










