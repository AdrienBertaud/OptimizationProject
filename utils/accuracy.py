import torch

def eval_accuracy(model, loss_function, data_loader):
    model.eval()
    n_batchs = len(data_loader)
    data_loader.idx = 0

    loss, acc = 0.0, 0.0

    with torch.no_grad():

        for i in range(n_batchs):
            inputs,targets = next(data_loader)

            targets_indices = torch.argmax(targets,1)

            y_hat = model(inputs)
            loss += loss_function(y_hat,targets_indices)
            acc += accuracy(y_hat,targets_indices)

    return loss/n_batchs, acc/n_batchs

def accuracy(y_hat, targets):

    y_hat_indices = torch.argmax(y_hat,1)
    acc = (targets==y_hat_indices).float()

    return torch.mean(acc)*100.0