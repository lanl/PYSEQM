import torch
import numpy as np
#np.set_printoptions(threshold=np.inf)

debug = True

def check_gradient(x,tag='tag'):
    if torch.isnan(x).any():
        print(x.detach().cpu().data.numpy())
        print(tag+": nan in gradient")
        raise ValueError(tag+": nan in gradient")
    if torch.isinf(x).any():
        print(x.detach().cpu().data.numpy())
        print(tag+": inf in gradient")
        raise ValueError(tag+": inf in gradient")

def check(x,tag='tag'):
    if debug:
        if torch.isnan(x).any():
            print(x.detach().cpu().data.numpy())
            print(tag+": nan in tensor")
            raise ValueError(tag+": nan in tensor")
        if torch.isinf(x).any():
            print(x.detach().cpu().data.numpy())
            print(tag+": inf in tensor")
            raise ValueError(tag+": inf in tensor")
        if x.requires_grad:
            x.register_hook(lambda grad: check_gradient(grad, tag=tag))
    else:
        pass

def save(x,name='name'):
    np.save(name+".npy",x.detach().cpu().data.numpy())

def check_dist(x, tag="name"):
    if debug:
        with torch.no_grad():
            print(tag + " abs max: " + str( x.abs().max(dim=0)[0]))
            print(tag + " mean   : " + str( x.mean(dim=0)))
            print(tag + " std    : " + str( x.std(dim=0)))
    else:
        pass
