import torch
import ot
    
def sinkhorn(a,b,C,lam=50,eps=1,numiter=500,lam2=None,pen=None,Kx=None,Ky=None):
    u=a
    G=torch.exp(-C/eps)
    if lam2 is None:
        lam2=lam
        
    if pen == "kkl":
        Lx = torch.linalg.cholesky(lam*Kx+eps*torch.eye(Kx.shape[0]))# Inverse with QR decomposition
        Ly = torch.linalg.cholesky(lam2*Ky+eps*torch.eye(Ky.shape[0]))
        Kxinv=torch.cholesky_inverse(Lx)
        Kyinv=torch.cholesky_inverse(Ly)
        Kxloga=Kx@torch.log(a)
        Kylogb=Ky@torch.log(b)
	    
    for i in range(numiter):
        if pen is None:
            prox=b
        elif pen=="kl":
            prox=prox_KL(G.T@u,b,lam2,eps)
        elif pen=="kkl":
            prox=prox_KKL(G.T@u,Kylogb,lam2,eps,Kyinv)
                                   
        v=prox/(G.T@u)
        
        if pen is None:
            prox=a
        elif pen=="kl":
            prox=prox_KL(G@v,a,lam,eps)
        elif pen=="kkl":
            prox=prox_KKL(G@v,Kxloga,lam,eps,Kxinv)
  
        u=prox/(G@v)
    

    return u.reshape((-1, 1)) * G * v.reshape((1, -1))


      
def prox_KL(Gu,b,lam,eps):
    gam1=lam/(eps+lam)
    gam2=eps/(eps+lam)
    return (Gu**gam2)*(b**gam1)

def prox_KKL(Gu,Klogb,lam,eps,Kinv):
    return torch.exp(Kinv@(eps*torch.log(Gu)+lam*Klogb))    

def prox_l2(Gv,b,lam):
    return (torch.real(lambertw(lam*Gv*torch.exp(lam*b), k=0, tol=1e-8))/lam).float()

def prox_MMD(Gv,b,K,lam,eps,u_warm=None,numiter=100,lr=1e-3):
    if u_warm is None:# warmstart
        u_warm=Gv
    u,loss_l=logGD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter,lr)
    #pl.plot(loss_l)
    return u
def MMDpen(u,Gv,b,K,lam,eps):
    return eps*(torch.sum(u*torch.log(u/Gv)-u+Gv))+lam*(b@K@b+u@K@u-2*b@K@u)


def loggrad_MMD(f,Gv,b,K,lam,eps):
    return eps*(f-torch.log(Gv))+2*lam*K@(torch.exp(f)-b)
  
def logGD_MMDpen(Gv,b,K,lam,eps,u_warm,numiter=100,lr=1e-2):
    f=torch.log(u_warm)
    loss_l=[]
    for i in range(numiter):
        f_next=f-lr*loggrad_MMD(f,Gv,b,K,lam,eps)
        f=f_next.clone()
        loss_l+=[MMDpen(torch.exp(f),Gv,b,K,lam,eps)]
    return torch.exp(f),loss_l
        
def prox_sinkhorn(Gv,b,G,lam,eps2,numiter,u): 
    if u is None:# warmstart
        u=Gv
    gamma=lam/(eps2+lam)
    u_old=u
    v_old=b
    for i in range(numiter):
        v=b/(G@u)
        u=(Gv/(G@v))**gamma
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<1e-5:
            v=b/(G@u)
            return u,v
        else:
            u_old=u
            v_old=v
    v=b/(G@u)
    return u,v
    
def prox_UOT_sinkhorn(Gv,b,G,lam1,eps2,numiter,lam2,u):
    if u is None:
        u=Gv
    gamma1=lam1/(eps2+lam1)
    gamma2=lam2/(eps2+lam2)
    u_old=u
    v_old=b
    for i in range(numiter):
        v=(b/(G@u))**gamma2
        u=(Gv/(G@v))**gamma1
        if (torch.norm(u_old-u)+torch.norm(v-v_old))<1e-5:
            v=b/(G@u)
            return u,v
        else:
            u_old=u
            v_old=v
    return u,v

    
