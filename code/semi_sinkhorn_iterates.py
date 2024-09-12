import torch
import ot
    

def semi_sinkhorn(a,b,C,lam2=50,eps=1,numiter=500,pen=None,Ky=None):
    u=a
    G=torch.exp(-C/eps)
    if lam2 is None:
        lam2=lam
        
    if pen == "kkl":
        Ly = torch.linalg.cholesky(lam2*Ky+eps*torch.eye(Ky.shape[0]))
        Kyinv=torch.cholesky_inverse(Ly)
        Kylogb=Ky@torch.log(b)
	  
    for i in range(numiter):
        if pen is None:
            prox=b
        elif pen=="kl":
            prox=prox_KL(G.T@u,b,lam2,eps)
        elif pen=="kkl":
            prox=prox_KKL(G.T@u,Kylogb,lam2,eps,Kyinv)     
        v=prox/(G.T@u)
        
        u=a/(G@v)
    
    return u.reshape((-1, 1)) * G * v.reshape((1, -1))

      
def prox_KL(Gu,b,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return (Gu**gam2)*(b**gam1)

def prox_KKL(Gu,Klogb,lam,eps,Kinv):
    return torch.exp(Kinv@(eps*torch.log(Gu)+lam*Klogb))   

