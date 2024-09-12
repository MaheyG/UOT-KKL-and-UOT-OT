import ot
import torch
import numpy as np

def factored_coupling(X, Y, r=100, Xb=None, stopThr=1e-8, numItermax=100):
    n = X.shape[0]
    d = X.shape[1]

    a,b=torch.ones((n,))/n,torch.ones((n,))/n

    if Xb is None:
        Xb = torch.randn(r, d)
        Xb = X[:r,:]

    w = torch.ones((r,)) / r

    def solve_ot(X, Y, a, b):
        return ot.emd(a,b,ot.dist(X,Y)).float()

    norm_delta = []

    # solve the barycenter
    for i in range(numItermax):

        old_Xb = Xb
        
        """pl.figure(figsize=(3,3))
        pl.scatter(X[:, 0], X[:, 1], c='C0', label='Source')
        pl.scatter(Y[:, 0], Y[:, 1], c='C1', label='Target')
        pl.scatter(Xb[:, 0], Xb[:, 1], c='C2', label='Target')"""
        # solve OT with template
        Ps = solve_ot(Xb, X, w, a)
        Pt = solve_ot(Xb, Y, w, b)
   
        Xb = 0.5 * (torch.matmul(Ps, X) + torch.matmul(Pt, Y)) * r

        delta = torch.norm(Xb - old_Xb)
                   
        if delta < stopThr:
            break

    return Ps.T,Pt,Xb
    
def factored_coupling_blur(X,Y,Ps,Pt,r):
    C=ot.dist(X,Y)
    P=r*(Ps@Pt)
    return P
    
    
def extract_factored(X,Y,Ps,Pt,r):
    n=X.shape[0]
    X_l=[]
    Y_l=[]
    
    for i in range(r):
        X_l.append(X[Ps[:,i]>=1/(n+1)])
        Y_l.append(Y[Pt[i,:]>=1/(n+1)])
    return X_l,Y_l
    
    
def factored_coupling_best(X,Y,Ps,Pt,r):
    X_l,Y_l=extract_factored(X,Y,Ps,Pt,r)
    
    W_l=[]
    for i in range(len(X_l)):
        C=ot.dist(X_l[i],Y_l[i])
        W_l=W_l+[ot.emd([],[],C.numpy())]
    
    return X_l,Y_l,W_l
    
def factored_coupling_worst(X,Y,Ps,Pt,r):
    X_l,Y_l=extract_factored(X,Y,Ps,Pt,r)
    
    W=0
    for i in range(len(X_l)):
        C=-ot.dist(X_l[i],Y_l[i])
        W+=ot.emd2([],[],C.numpy())/r
    
    return -W
    
def factored_coupling_random(X,Y,Ps,Pt,r):
    n=X.shape[0]
    
    X_l,Y_l=extract_factored(X,Y,Ps,Pt,r)
    p=X_l[0].shape[0]

    W_l=[]
    for i in range(len(X_l)):
        W_l=W_l+[np.random.permutation(np.eye(p))]
        #W_l=W_l+[torch.randperm(torch.eye(p))]
    return X_l,Y_l,W_l

