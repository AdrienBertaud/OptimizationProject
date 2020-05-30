import torch
import numpy as np
from imag_processing import label_to_img



def get_prediction(net,test_iter,window_size,stride,IMG_PATCH_SIZE, test_w, test_h, device):  #For prediction, stride need to be 16
    """
    Generate mask image for prediction given device type, window size, test image size and specified model.
    For prediction, stride need set to 16.
    """

    Y_np = np.empty([(test_w//IMG_PATCH_SIZE)*(test_h//IMG_PATCH_SIZE)])
    i = 0
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        for X in test_iter:
            X = X.to(device)
            Yscore_ts = net(X)
            Y_ts_buf = Yscore_ts.argmax(dim = 1)
            #Y_ts = torch.cat((Y_ts,Y_ts_buf), dim = 0)
            Y_np_buf = Y_ts_buf.cpu().numpy()
            Y_np[i:i+len(X)] = Y_np_buf
            #print(Y_np[i:i+len(X)])
            i = i+len(X)

    img_prediction = label_to_img(test_w, test_h, IMG_PATCH_SIZE, IMG_PATCH_SIZE, Y_np)

    return img_prediction