import math
import src.linalg
from importlib import reload
reload(src.linalg)
from src.linalg import eigen_variance, eigen_hessian

def diagnose(net, criterion, optimizer, train_loader,test_loader, n_iters=10, tol=1e-4, verbose=True):

    print('===> Compute sharpness on train data:')
    sharpness_train = get_sharpness(net, criterion, train_loader, \
                                n_iters=n_iters, tol=tol, verbose=verbose)
    print('Sharpness is %.2e\n'%(sharpness_train))

    print('===> Compute sharpness on test data:')
    sharpness_test = get_sharpness(net, criterion, test_loader, \
                                n_iters=n_iters, tol=tol, verbose=verbose)
    print('Sharpness is %.2e\n'%(sharpness_test))

    print('===> Compute non-uniformity on train data:')
    non_uniformity_train = get_nonuniformity(net, criterion, train_loader, \
                                        n_iters=n_iters, tol=tol, verbose=verbose)
    print('Non-uniformity is %.2e\n'%(non_uniformity_train))

    print('===> Compute non-uniformity on test data:')
    non_uniformity_test = get_nonuniformity(net, criterion, test_loader, \
                                        n_iters=n_iters, tol=tol, verbose=verbose)
    print('Non-uniformity is %.2e\n'%(non_uniformity_test))

    print("sharpness train = ", sharpness_train)
    print("non uniformity train = ", non_uniformity_train)
    print("sharpness test = ", sharpness_test)
    print("non uniformity test = ", non_uniformity_test)

    return sharpness_train, non_uniformity_train, sharpness_test, non_uniformity_test


def get_sharpness(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_hessian(net, criterion, dataloader, \
                      n_iters=n_iters, tol=tol, verbose=verbose)
    return v


def get_nonuniformity(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_variance(net, criterion, dataloader, \
                      n_iters=n_iters, tol=tol, verbose=verbose)
    if v<0:
        print("ERROR: eigen variance is negative : ", v)
        return 0
    else:
        return math.sqrt(v)

