B = 3
T = 7
Z = 1
T2 = 10

mean = torch.randn(B,T,Z)
var = torch.ones(B,T,Z)*0.5
mixd = torch.distributions.Categorical(torch.ones(B,T))
compd = torch.distributions.Independent(torch.distributions.Normal(mean, var),1)
gmm = torch.distributions.MixtureSameFamily(mixd, compd)

mean2 = mean.clone().detach()
mean2[1,4] = -mean2[1,4]
compd2 = torch.distributions.Independent(torch.distributions.Normal(mean2, var),1)
gmm2 = torch.distributions.MixtureSameFamily(mixd, compd2)

x = torch.randn(T,B,Z)
y = torch.randn(T2,B,Z)

gmm.log_prob(x) 
gmm2.log_prob(x)

gmm.log_prob(y)
gmm2.log_prob(y)

################################################################################################
################################################################################################
# Different dimensions
################################################################################################
################################################################################################

B = 3
T = 7
Z = 2
T2 = 10

mean = torch.randn(B,T,Z)
var = torch.ones(B,T,Z)*0.5
mixd = torch.distributions.Categorical(torch.ones(B,T))
compd = torch.distributions.Independent(torch.distributions.Normal(mean, var),1)
gmm = torch.distributions.MixtureSameFamily(mixd, compd)

mean2 = mean.clone().detach()
mean2[1,4] = -mean2[1,4]
mean2[1,6] = -mean2[1,6]
compd2 = torch.distributions.Independent(torch.distributions.Normal(mean2, var),1)
gmm2 = torch.distributions.MixtureSameFamily(mixd, compd2)

x = torch.randn(T,B,Z)
y = torch.randn(T2,B,Z)

#########
Flip

B = 2
T = 5
Z = 1
T2 = 8

mean = torch.randn(T,B,Z)
var = torch.ones(T,B,Z)*0.5
mixd = torch.distributions.Categorical(torch.ones(T,B))
compd = torch.distributions.Independent(torch.distributions.Normal(mean, var),1)
gmm = torch.distributions.MixtureSameFamily(mixd, compd)

mean2 = mean.clone().detach()
mean2[3,1] = -mean2[3,1]
compd2 = torch.distributions.Independent(torch.distributions.Normal(mean2, var),1)
gmm2 = torch.distributions.MixtureSameFamily(mixd, compd2)

x = torch.randn(B,T,Z)
y = torch.randn(B,T2,Z)

gmm.log_prob(x) 
gmm2.log_prob(x)

gmm.log_prob(y)
gmm2.log_prob(y)