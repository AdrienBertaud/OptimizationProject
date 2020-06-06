
from src.utils import get_sharpness, get_nonuniformity

def diagnose(net, criterion, optimizer, train_loader,test_loader, n_iters=10, tol=1e-4, verbose=True):

    print('===> Compute sharpness on train data:')
    sharpness = get_sharpness(net, criterion, train_loader, \
                                n_iters=n_iters, tol=tol, verbose=verbose)
    print('Sharpness is %.2e\n'%(sharpness))

    print('===> Compute sharpness on test data:')
    sharpness = get_sharpness(net, criterion, test_loader, \
                                n_iters=n_iters, tol=tol, verbose=verbose)
    print('Sharpness is %.2e\n'%(sharpness))

    print('===> Compute non-uniformity on train data:')
    non_uniformity = get_nonuniformity(net, criterion, train_loader, \
                                        n_iters=n_iters, tol=tol, verbose=verbose)
    print('Non-uniformity is %.2e\n'%(non_uniformity))

    print('===> Compute non-uniformity on test data:')
    non_uniformity = get_nonuniformity(net, criterion, test_loader, \
                                        n_iters=n_iters, tol=tol, verbose=verbose)
    print('Non-uniformity is %.2e\n'%(non_uniformity))

    return sharpness, non_uniformity

