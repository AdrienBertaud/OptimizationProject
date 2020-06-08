import math
import utils.linalg
from importlib import reload
reload(utils.linalg)
from utils.linalg import eigen_variance, eigen_hessian


def diagnose(net, criterion, optimizer, data_loader):

    n_iters=10
    tol=1e-4

    print('Compute sharpness :')
    sharpness = get_sharpness(net, criterion, data_loader, \
                              n_iters=n_iters, tol=tol)
    print('Sharpness is %.2e\n'%(sharpness))

    print('Compute non-uniformity :')
    non_uniformity = get_nonuniformity(net, criterion, data_loader, \
                                       n_iters=n_iters, tol=tol)
    print('Non-uniformity is %.2e\n'%(non_uniformity))

    return sharpness, non_uniformity


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

