import os
import time
import argparse
import json
import torch

from src.utils import load_net, load_data, \
                      get_sharpness, get_nonuniformity, \
                      eval_accuracy

def get_args():

    optimizerName = 'sgd'

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',default='0,')
    argparser.add_argument('--dataset',default='fashionmnist',
                            help='dataset choosed, [fashionmnist] | cifar10')
    # argparser.add_argument('--n_samples',type=int,
    #                         default=1000, help='training set size, [1000]')
    # argparser.add_argument('--batch_size', type=int,
    #                         default=1000, help='batch size')
    argparser.add_argument('--model_file', default=optimizerName+'.pkl',
                            help='file name of the pretrained model')
    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

    print('===> Config:')
    print(json.dumps(vars(args),indent=2))
    return args

def diagnose(optimizerName, n_iters=10, n_samples=1000, batch_size=1000):
    args = get_args()

    args = get_args()
    # load model
    #criterion = torch.nn.MSELoss().cuda()
    criterion = torch.nn.MSELoss()
    train_loader,test_loader = load_data(args.dataset,
                                        training_size=n_samples,
                                        batch_size=batch_size)
    net = load_net(args.dataset)
    net.load_state_dict(torch.load(optimizerName))

    # Evaluate models
    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)

    print('===> Basic information of the given model: ')
    print('\t train loss: %.2e, acc: %.2f'%(train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f'%(test_loss, test_accuracy))

    print('===> Compute sharpness:')
    sharpness = get_sharpness(net, criterion, train_loader, \
                                n_iters=n_iters, verbose=True, tol=1e-4)
    print('Sharpness is %.2e\n'%(sharpness))

    print('===> Compute non-uniformity:')
    non_uniformity = get_nonuniformity(net, criterion, train_loader, \
                                        n_iters=n_iters, verbose=True, tol=1e-4)
    print('Non-uniformity is %.2e\n'%(non_uniformity))

    return sharpness, non_uniformity

def main():
    diagnose(get_args().model_file)

if __name__ == '__main__':
    main()
