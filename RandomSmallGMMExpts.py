# coding: utf-8
import torch
torch.nn.Linear?
x = torch.randn(5).to(device)
x = torch.randn(5).cuda()
torch.autograd.Variable?
y = torch.autograd.Variable(torch.ones(5),requires_grad=True).cuda()
opt = torch.optim.Adam(1e-4, y)
y
y.parameters
y.parameters()
opt = torch.optim.Adam([y],lr=1)
y
y
y = torch.autograd.Variable(torch.ones(5),requires_grad=True).cuda()
y
clear
opt = torch.optim.Adam(y,lr=1)
opt = torch.optim.Adam([y],lr=1)
y = torch.autograd.Variable(torch.ones(5),requires_grad=True,device='cuda')
y = torch.ones(5,requires_grad=True,device='cuda')
y
opt = torch.optim.Adam([y],lr=1)
x = torch.randn((5,4)).cuda()
y = torch.randn((5,2),requires_grad=True,device='cuda')
y = torch.randn((2,4),requires_grad=True,device='cuda')
x = torch.randn((2,4)).cuda()
y = torch.randn((5,4),requires_grad=True,device='cuda')
opt = torch.optim.Adam([y],lr=1)
opt.zero_grad()
mixd = torch.distributions.Categorical
mixd = torch.distributions.Categorical(torch.ones(2).cuda())
comd = torch.distributions.Independent(torch.distributions.Normal(x,torch.ones_like(x).cuda()),1)
gmm = torch.distributions.MixtureSameFamily(mixd, comd)
gmm.log_prob(y)
y
for k in range(100):
    opt.zero_grad()
    loss = - gmm.log_prob(y) 
    loss.sum().backward()
    opt.step()
    print('iter',k)
    print('y:',y)
x
x.mean(dim=1)
x.mean(dim=0)
clear
x.mean(dim=0)
x = torch.randn((2,4)).cuda()
y = torch.randn((5,4),requires_grad=True,device='cuda')
opt = torch.optim.Adam([y],lr=1)
mixd = torch.distributions.Categorical(torch.ones(2).cuda())
comd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
gmmv = 0.01
comd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
gmm = torch.distributions.MixtureSameFamily(mixd, comd)
gmm.log_prob(y)
for k in range(1000):
    opt.zero_grad()
    loss = - gmm.log_prob(y) 
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
x
x.mean()
x.mean(dim=0)
y
clear
x = torch.randn((4,4)).cuda()
y = torch.randn((5,4),requires_grad=True,device='cuda')
x
y
opt = torch.optim.Adam([y],lr=1)
for k in range(1000):
    opt.zero_grad()
    loss = - gmm.log_prob(y) 
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
x
clear
x = torch.randn((4,4)).cuda()
y = torch.randn((5,4),requires_grad=True,device='cuda')
mixd = torch.distributions.Categorical(torch.ones(2).cuda())
comd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
gmm = torch.distributions.MixtureSameFamily(mixd, comd)
mixd = torch.distributions.Categorical(torch.ones(N).cuda())
x = torch.randn((N,4)).cuda()
N = 4
x = torch.randn((N,4)).cuda()
mixd = torch.distributions.Categorical(torch.ones(N).cuda())
comd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
gmm = torch.distributions.MixtureSameFamily(mixd, comd)
opt = torch.optim.Adam([y],lr=1)
for k in range(1000):
    opt.zero_grad()
    loss = - gmm.log_prob(y) 
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
y
x
fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
x = torch.randn((N,4)).cuda()
M = 5
y = torch.randn((M,4),requires_grad=True,device='cuda')
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
ggmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)
opt = torch.optim.Adam([y],lr=1)
for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y) 
    bloss = - bgmm.log_prob(x)
    loss = bloss+floss
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
bloss
floss
for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y) 
    bloss = - bgmm.log_prob(x)
    loss = bloss.sum()+floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
x
y
x = torch.randn((N,4)).cuda()
y = torch.randn((M,4),requires_grad=True,device='cuda')
fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)
for k in range(1000):
    opt.zero_grad()
    # floss = - fgmm.log_prob(y) 
    floss = 0.
    bloss = - bgmm.log_prob(x)
    loss = bloss.sum()+floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y) 
    
    bloss = - bgmm.log_prob(x)
    # loss = bloss.sum()+floss.sum()
    loss = bloss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
y
x
z = torch.randn((M,4),requires_grad=True,device='cuda')
clear
x = torch.randn((N,4)).cuda()
y = z.clone()
clear
x = torch.randn((N,4)).cuda()
z = torch.randn((M,4),requires_grad=True,device='cuda')
y = z.detach().clone()
y
y.requires_grad
y.requires_grad = True
y
fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)
for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y) 
    
    bloss = - bgmm.log_prob(x)
    # loss = bloss.sum()+floss.sum()
    loss = bloss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
x
y = z.detach().clone()
z
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)
opt = torch.optim.Adam([y],lr=1)


for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y) 
    
    bloss = - bgmm.log_prob(x)
    # loss = bloss.sum()+floss.sum()
    # loss = bloss.sum()
    loss = floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)
z = torch.randn((M,4),requires_grad=True,device='cuda')
z
ls
%?
%save?



import torch

def run_expt():
    N = 4
    M = 5
    a = torch.randn((N,4)).cuda()
    b = torch.randn((M,4),requires_grad=True,device='cuda')
    x = a.clone()
    y = b.clone()

    opt = torch.optim.Adam([y],lr=1)
    fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
    fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
    fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
    bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
    bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
    bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

    for k in range(1000):
        opt.zero_grad()
        floss = - fgmm.log_prob(y) 
        
        bloss = - bgmm.log_prob(x)
        # loss = bloss.sum()+floss.sum()
        # loss = bloss.sum()
        loss = floss.sum()
        loss.sum().backward()
        opt.step()
        if k%100==0:
            print('iter',k)
            print('y:',y)


#
import torch
import numpy as np

N = 3
M = 5
a = np.random.randn(N,4)
b = np.random.randn(M,4)

# 

# x = torch.randn((N,4)).cuda()
# y = torch.randn((M,4),requires_grad=True,device='cuda')
# x.data = a
# y.data = b

# 
x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

gmmv = 0.05
opt = torch.optim.Adam([y],lr=1)
fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y)     
    bloss = - bgmm.log_prob(x)
    loss = bloss.sum()+floss.sum()
    # loss = bloss.sum()
    # loss = floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)

############################################################################
# Non sequential setting, but recreate reverse GMM at every step? Do we need this? or same result? 
# Test

# First the version where we don't recreate the GMM..

import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False, precision=4)


N = 3
M = 5
a = np.random.randn(N,4)
b = np.random.randn(M,4)

# x = torch.randn((N,4)).cuda()
# y = torch.randn((M,4),requires_grad=True,device='cuda')
# x.data = a
# y.data = b

# 
x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)
fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

for k in range(1000):
    opt.zero_grad()     
    floss = - fgmm.log_prob(y)     
    bloss = - bgmm.log_prob(x)
    # loss = bloss.sum()+floss.sum()
    loss = bloss.sum()
    # loss = floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k,'loss',loss)
        print('y:',y)        

######################################################
######################################################
# Test with a bunch of gmmv values... 
gmmvl = [0.0001, 0.001, 0.01, 0.1, 1.]
for gmmv in gmmvl:
    print("################################################################################################")
    print("################################################################################################")
    print("################################################################################################")
    x = torch.from_numpy(a).cuda()
    y = torch.from_numpy(b).cuda()
    y.requires_grad=True

    # gmmv = 0.001
    opt = torch.optim.Adam([y],lr=1)
    fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
    fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
    fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
    bmixd = torch.distributions.Categorical(torch.ones(M).cuda())
    bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
    bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

    for k in range(1000):
        opt.zero_grad()     
        floss = - fgmm.log_prob(y)     
        bloss = - bgmm.log_prob(x)
        # loss = bloss.sum()+floss.sum()
        loss = bloss.sum()
        # loss = floss.sum()
        loss.sum().backward()
        opt.step()
        if k%100==0:
            print('iter',k,'gmmv',gmmv,'loss',loss)
            print('y:',y)        

######################################################
######################################################

######################################################
# Now the version where we DO recreate the GMM..
######################################################

import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False, precision=4)


N = 3
M = 5
a = np.random.randn(N,4)
b = np.random.randn(M,4)

# x = torch.randn((N,4)).cuda()
# y = torch.randn((M,4),requires_grad=True,device='cuda')
# x.data = a
# y.data = b

# 
x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)
fmixd = torch.distributions.Categorical(torch.ones(N).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bmixd = torch.distributions.Categorical(torch.ones(M).cuda())

for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y)

    bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
    bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)
     
    bloss = - bgmm.log_prob(x)
    loss = bloss.sum()+floss.sum()
    # loss = bloss.sum()
    # loss = floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)

############################################################################
############################################################################
# Sequential version of this experiment. 
############################################################################
############################################################################
import torch
import numpy as np

N = 3
M = 3
T = 5
a = np.random.randn(N,T,4)
b = np.random.randn(M,T,4)

######################################
# Non-sequential baseline
######################################
x = torch.from_numpy(a.reshape(-1,4)).cuda()
y = torch.from_numpy(b.reshape(-1,4)).cuda()
y.requires_grad=True

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)
fmixd = torch.distributions.Categorical(torch.ones(N*T).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bmixd = torch.distributions.Categorical(torch.ones(M*T).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y)     
    bloss = - bgmm.log_prob(x)
    # loss = bloss.sum()+floss.sum()
    # loss = bloss.sum()
    loss = floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)

######################################
# Actual sequential set based thing.
######################################

import torch
import numpy as np

N = 4
M = 4
T = 5
a = np.random.randn(N,T,4)
b = np.random.randn(M,T,4)

######################################

x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)

fmixd = torch.distributions.Categorical(torch.ones(N*T).cuda())
fcomd = torch.distributions.Independent(torch.distributions.Normal(x,gmmv*torch.ones_like(x).cuda()),1)
fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
bmixd = torch.distributions.Categorical(torch.ones(M*T).cuda())
bcomd = torch.distributions.Independent(torch.distributions.Normal(y,gmmv*torch.ones_like(y).cuda()),1)
bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

for k in range(1000):
    opt.zero_grad()
    floss = - fgmm.log_prob(y)     
    bloss = - bgmm.log_prob(x)
    # loss = bloss.sum()+floss.sum()
    # loss = bloss.sum()
    loss = floss.sum()
    loss.sum().backward()
    opt.step()
    if k%100==0:
        print('iter',k)
        print('y:',y)

######################################


x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)

for i in range(M):

    print("#########################")
    print("Remember, X[i] is:", x[i])
    print("Remember, B[i] is:", b[i])
    # For every pair of datapoints
    fmixd = torch.distributions.Categorical(torch.ones(T).cuda())
    fcomd = torch.distributions.Independent(torch.distributions.Normal(x[i],gmmv*torch.ones_like(x[i]).cuda()),1)
    fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
    bmixd = torch.distributions.Categorical(torch.ones(T).cuda())
    bcomd = torch.distributions.Independent(torch.distributions.Normal(y[i],gmmv*torch.ones_like(y[i]).cuda()),1)
    bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

    for k in range(1000):
        opt.zero_grad()
        floss = - fgmm.log_prob(y[i])     
        bloss = - bgmm.log_prob(x[i])
        loss = bloss.sum()+floss.sum()
        # loss = bloss.sum()
        # loss = floss.sum()
        loss.sum().backward()
        opt.step()
        if (k+1)%1000==0:
            print('iter',k)
            print('y:',y[i])


######################################
# GMM over Z tuples...
######################################


import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False, precision=4)

N = 4
M = 4
T = 5
a = np.random.randn(N,T,4)
b = np.random.randn(M,T,4)


x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

zi = [[i,i+1] for i in range(T-1)]

# xtups = [x[:,zi[k]].view(-1,8) for k in range(T-1)]
# ytups = [y[:,zi[k]].view(-1,8) for k in range(T-1)]

xtups = [torch.cat([x[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]
ytups = [torch.cat([y[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)

for i in range(M):

    print("#########################")
    print("Remember, X[i] is:", x[i])
    print("Remember, B[i] is:", b[i])
    # For every pair of datapoints
    fmixd = torch.distributions.Categorical(torch.ones(T-1).cuda())
    # fcomd = torch.distributions.Independent(torch.distributions.Normal(x[i],gmmv*torch.ones_like(x[i]).cuda()),1)
    fcomd = torch.distributions.Independent(torch.distributions.Normal(xtups[i],gmmv*torch.ones_like(xtups[i]).cuda()),1)
    fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
    bmixd = torch.distributions.Categorical(torch.ones(T-1).cuda())
    # bcomd = torch.distributions.Independent(torch.distributions.Normal(ytups[i],gmmv*torch.ones_like(ytups[i]).cuda()),1)
    # bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

    for k in range(1000):
        opt.zero_grad()
        ytups = [torch.cat([y[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]
        bcomd = torch.distributions.Independent(torch.distributions.Normal(ytups[i],gmmv*torch.ones_like(ytups[i]).cuda()),1)
        bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

        floss = - fgmm.log_prob(ytups[i])     
        bloss = - bgmm.log_prob(xtups[i])
        loss = bloss.sum()+floss.sum()
        # loss = bloss.sum()
        # loss = floss.sum()
        loss.backward()
        opt.step()
        if (k+1)%1000==0:
            print('iter',k)
            print('y:',y[i])


################################
# Test CDSL
################################

import torch
import numpy as np
np.set_printoptions(suppress=True, precision=4)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False, precision=4)

N = 4
M = 4
T = 5
a = np.random.randn(N,T,4)
b = np.random.randn(M,T,4)

x = torch.from_numpy(a).cuda()
y = torch.from_numpy(b).cuda()
y.requires_grad=True

zi = [[i,i+1] for i in range(T-1)]

# xtups = [x[:,zi[k]].view(-1,8) for k in range(T-1)]
# ytups = [y[:,zi[k]].view(-1,8) for k in range(T-1)]

xtups = [torch.cat([x[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]
ytups = [torch.cat([y[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]

gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)

for i in range(M):

    print("#########################")
    print("Remember, X[i] is:", x[i])
    print("Remember, B[i] is:", b[i])
    # For every pair of datapoints

    for k in range(1000):
        opt.zero_grad()
        ytups = [torch.cat([y[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]
        loss = ((ytups[i]-xtups[i])**2).mean()
        loss.backward()
        opt.step()
        if (k+1)%1000==0:
            print('iter',k)
            print('y:',y[i])

######################################################################
######################################################################


import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True, precision=4)

# Generate some data to test on... 
bs1 = np.load("B_GMM_Toy.npy")
ndp = 1
nt = 19
T = 20

zs1 = torch.zeros((ndp,nt,4)).cuda()
for i in range(ndp):
    for t in range(nt):
        if bs1[i,t]==1:
            zs1[i,t] = torch.randn(4).cuda()
        else:
            zs1[i,t] = zs1[i,t-1]

bs2 = np.load("B_GMM_Toy2.npy")
zs2 = torch.zeros((ndp,nt,4)).cuda()
for i in range(ndp):
    for t in range(nt):
        if bs2[i,t]==1:
            zs2[i,t] = torch.randn(4).cuda()
        else:
            zs2[i,t] = zs2[i,t-1]

# x = torch.from_numpy(zs1).cuda()
# y = torch.from_numpy(zs2).cuda()
zs1_copy = zs1.clone().detach()
zs2_copy = zs2.clone().detach()
x = zs1
y = zs2

y.requires_grad=True

# Create tuples..
# Dom1

def settuples(b,x):
    tup = [torch.zeros((b[i].sum()-1,8)).cuda() for i in range(ndp)]

    for i in range(ndp):
        # zs1[i,np.where(bs1[i])[0]].view(-1,8)
        bv = np.where(b[i])[0]
        for k,v in enumerate(bv[:-1]):        
            tup[i][k] = x[i,[bv[k],bv[k+1]]].view(-1,8)  

    return tup

xtups = settuples(bs1,x)
ytups = settuples(bs2,y)
T = 20
gmmv = 0.01
opt = torch.optim.Adam([y],lr=1)

for i in range(ndp):

    print("#########################")
    print("Remember, X[i] is:", x[i])
    # print("Remember, B[i] is:", bs1[i])
    # For every pair of datapoints
    fmixd = torch.distributions.Categorical(torch.ones(xtups[i].shape[0]).cuda())
    # fcomd = torch.distributions.Independent(torch.distributions.Normal(x[i],gmmv*torch.ones_like(x[i]).cuda()),1)
    fcomd = torch.distributions.Independent(torch.distributions.Normal(xtups[i],gmmv*torch.ones_like(xtups[i]).cuda()),1)
    fgmm = torch.distributions.MixtureSameFamily(fmixd, fcomd)
    
    bmixd = torch.distributions.Categorical(torch.ones(ytups[i].shape[0]).cuda())
    # bcomd = torch.distributions.Independent(torch.distributions.Normal(ytups[i],gmmv*torch.ones_like(ytups[i]).cuda()),1)
    # bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

    for k in range(1000):
        opt.zero_grad()
        # ytups = [torch.cat([y[i][zi[k]].view(-1,8) for k in range(T-1)]) for i in range(M)]

        ytups = settuples(bs2,y)
        bcomd = torch.distributions.Independent(torch.distributions.Normal(ytups[i],gmmv*torch.ones_like(ytups[i]).cuda()),1)
        bgmm = torch.distributions.MixtureSameFamily(bmixd, bcomd)

        floss = - fgmm.log_prob(ytups[i])     
        bloss = - bgmm.log_prob(xtups[i])
        # loss = bloss.sum()+floss.sum()
        # loss = bloss.sum()
        loss = floss.sum()
        loss.backward()
        opt.step()
        if (k+1)%1000==0:
            print('iter',k)
            print('y:',y[i])
