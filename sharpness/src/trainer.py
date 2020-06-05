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

            # if optimizerName == 'gd':
            #     batch_size_used = dataloader.n_samples
            # else:
            #     batch_size_used = batch_size

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
            break;

    return iter_now

def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss,acc = 0,0

    #inputs, targets = inputs.cuda(), targets.cuda()
    inputs,targets = next(dataloader)
    logits = model(inputs)
    

    for logit, target in zip(logits, targets):
        
        E = criterion(logit,target)
        E.backward(retain_graph=True)
    
        loss += E.item()
        acc += accuracy(logit.data,target)

    # TODO: ?
    for p in model.parameters():
        p.grad.data /= batch_size

    return loss/batch_size, acc/batch_size

def accuracy(logit, target):

    y_trues = torch.argmax(target)
    y_preds = torch.argmax(logit)
    acc = (y_trues==y_preds).float()*100.0
    
    return acc












