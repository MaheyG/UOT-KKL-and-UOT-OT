import torch
import numpy as np
import ot
from utils import kernel 
from sinkhorn_iterates import sinkhorn
from semi_sinkhorn_iterates import semi_sinkhorn
from semi_linear_solver import semi_UOT_KL,semi_UOT_MMD
import sklearn
from gwot import sim
import autograd.numpy as np
import autograd
import gwot

def traj_balanced(X,eps=1e-1,numiter=100):
    T,N,d=X.shape
    Y=torch.clone(X)  #EMD
    Ye=torch.clone(X) #Sinkhorn
    w=torch.ones((T,N)) #weights
    for i in range(T-1):
        #Linear solver
        C=ot.dist(Y[i,:,1:],Y[i+1,:,1:])
        P=ot.emd(w[i],w[i+1],C)
        Y[i+1,:,:]=P@Y[i+1,:,:]

        #Sinkhorn
        C=ot.dist(Ye[i,:,1:],Ye[i+1,:,1:])
        #u,v,G=semi_sinkhorn(torch.tensor(w1[i]),torch.tensor(w1[i+1]),C,eps=eps,numiter=numiter)
        #P=u.reshape((-1, 1)) * G * v.reshape((1, -1))
        P=sinkhorn(w[i],w[i+1],C,eps=eps,numiter=numiter)
        Ye[i+1,:,:]=P@Ye[i+1,:,:]
    return Ye,Y

def traj_UOT_KL(X,lam,eps=1e-1,numiter=100):
    T,N,d=X.shape
    Y=torch.clone(X)
    Ye=torch.clone(X)
    w=torch.ones((X.shape[0],X.shape[1]))
    for i in range(T-1):    
        #Linear solver
        C=ot.dist(Y[i,:,1:],Y[i+1,:,1:])
        P=semi_UOT_KL(w[i],w[i+1],C,lam2=lam,solver="CLARABEL")
        Y[i+1,:,:]=torch.tensor(P)@Y[i+1,:,:]

        #Sinkhorn iterates
        C=ot.dist(Ye[i,:,1:],Ye[i+1,:,1:])
        P=semi_sinkhorn(w[i],w[i+1],C,lam2=lam,eps=eps,numiter=numiter,pen="kl")
        Ye[i+1,:,:]=P@Ye[i+1,:,:]
        
    return Ye,Y
        
def traj_UOT_MMD(X,lam,sigma=1,regul=1e-1):
    T,N,d=X.shape
    Y=torch.clone(X)
    w=torch.ones((X.shape[0],X.shape[1]))
    for i in range(T-1):
        #Linear solver
        C=ot.dist(Y[i,:,1:],Y[i+1,:,1:])
        _,Ky=kernel(Y[i,:,1:],Y[i+1,:,1:],sigma=sigma,k="gaussian")
        P=semi_UOT_MMD(w[i],w[i+1],C,lam2=lam,Ky=Ky,solver="CLARABEL",regul=regul)
        
        Y[i+1,:,:]=torch.tensor(P)@Y[i+1,:,:]
    return Y
    
def traj_UOT_KKL(X,lam,eps=1e-1,sigma=1,numiter=100):
    T,N,d=X.shape
    Y=torch.clone(X)
    w=torch.ones((X.shape[0],X.shape[1]))
    for i in range(T-1):
        C=ot.dist(Y[i,:,1:],Y[i+1,:,1:])
        _,Ky=kernel(Y[i,:,1:],Y[i+1,:,1:],sigma=sigma,k="gaussian")
        P=semi_sinkhorn(w[i],w[i+1],C,lam2=lam,eps=eps,numiter=numiter,pen="kkl",Ky=Ky)
        Y[i+1,:,:]=P@Y[i+1,:,:]
    return Y
    
def dist_traj(X_true,X):
    T=X_true.shape[0]
    X_true=X_true.numpy()
    X=X.numpy()
    C=sklearn.metrics.pairwise_distances(np.transpose(X[:,:,1:],(1,0,2)).reshape(-1,T),
                                       np.transpose(X_true[:,:,1:],(1,0,2)).reshape(-1,T), metric = 'sqeuclidean')
    return ot.emd2([],[],C)


def Psi(x, t, dim = 4):
	    x0 = 1.4*np.array([1, 1] + [0, ]*(dim - 2))
	    x1 = -1.25*np.array([1, 1] + [0, ]*(dim - 2))
	    return 1.25*np.sum((x - x0)*(x - x0), axis = -1) * np.sum((x - x1)*(x - x1), axis = -1) + 10*np.sum(x[:, 2:]*x[:, 2:], axis = -1)
def sigmoide(x,tau=10):
    return 1/(1+np.exp(-tau*x))

def sigmoide2(x,tau=50):
    x=x-.5
    return 1/(1+np.exp(-tau*x))	 
       
def sampling(T,N,n,seed=1,shift=5,scale=.1,tau=30):
	dPsi = autograd.elementwise_grad(Psi)
	# setup simulation parameters
	sim_steps = 1000 # number of steps to use for Euler-Maruyama method
	D = 1 # diffusivity
	t_final = 1 # simulation run on [0, t_final]

	# branching rates
	R = 10
	beta = lambda x, t: R*((np.tanh(2*x[0]) + 1)/2)
	delta = lambda x, t: 0
	r = lambda x, t: beta(x, t) - delta(x, t)

	# function for particle initialisation
	ic_func = lambda N, d: np.random.randn(N, d)*0.1

	# setup simulation object
	sim = gwot.sim.Simulation(V = Psi, dV = dPsi, birth_death = True, birth = beta, death = delta,
		                  N = np.repeat(N, T), T = T, d = 4, D = D, t_final = t_final, ic_func = ic_func, pool = None)
	
	np.random.seed(seed)
	paths = sim.sample_trajectory(steps_scale = int(sim_steps/sim.T), N = n*N)
	X_true= np.transpose(paths,(1,0,2))
	X_true=X_true[:,:,0]
	S=np.random.normal(np.kron(np.max(X_true,1)+shift*sigmoide2(np.linspace(0,1,T),tau=tau),np.ones((int(n*N/3),1))).T,
		           scale=np.kron(scale*sigmoide(np.linspace(-1,1,T)),np.ones((int(n*N/3),1))).T)
	S-=np.random.normal(np.zeros(S.shape),scale=scale*np.ones(S.shape))
	X_true=np.column_stack((X_true,S))
	X_true=X_true.reshape(-1,1)
	X_true=np.column_stack((np.kron(np.linspace(0, 1*t_final, T), np.ones(N*n+int(n*N/3))),X_true))
	X_true=X_true.reshape(T,N*n+int(N*n/3),2)
	return torch.tensor(X_true)

def subsampling(X,N,n,seed=0):
    T=X.shape[0]
    np.random.seed(seed)
    X_sub=np.zeros((T,N+int(N/3),2))
    for i in range(T):
        idx=np.concatenate((np.ones(N+int(N/3)),np.zeros(n*N+int(n*N/3)-N-int(N/3))))
        np.random.shuffle(idx)
        X_sub[i,:,:]=X[i,idx.astype(bool),:]
    return torch.tensor(X_sub)

