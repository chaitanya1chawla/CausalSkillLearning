# Actually get Z's. Hopefully this goes over representative proportion of dataset.
self.source_manager.get_trajectory_and_latent_sets(get_visuals=False)
self.target_manager.get_trajectory_and_latent_sets(get_visuals=False)
# self.target_manager.get_trajectory_and_latent_sets(get_visuals=False, N=200)



###################################
# Create GMM
self.gmm_variance_value = 0.2
# Assumes self.source_z_GMM_component_means is of shape self.number_of_components x self.number of z dimensions.
# self.gmm_means = torch.tensor(self.source_z_GMM_component_means).to(device)
y = np.concatenate(self.target_manager.latent_z_set)[:500]
self.gmm_means = torch.tensor(y).to(device)

x = np.concatenate(self.source_manager.latent_z_set)[:500]
xt = torch.tensor(x).to(device)


self.gmm_variances = self.gmm_variance_value*torch.ones_like(self.gmm_means).to(device)
# self.mixture_distribution = torch.distributions.Categorical(torch.ones(self.number_of_components).to(device))
self.mixture_distribution = torch.distributions.Categorical(torch.ones(self.gmm_means.shape[0]).to(device))

self.component_distribution = torch.distributions.Independent(torch.distributions.Normal(self.gmm_means,self.gmm_variances),1)
self.GMM = torch.distributions.MixtureSameFamily(self.mixture_distribution, self.component_distribution)

# Reverse GMM
self.reverse_component_distribution = torch.distributions.Independent(torch.distributions.Normal(xt,self.gmm_variances),1)
self.reverese_GMM = torch.distributions.MixtureSameFamily(self.mixture_distribution, self.reverse_component_distribution)

###################################
###################################
#### Using translated z sets
###################################
###################################

self.gmm_means = torch.tensor(self.target_latent_zs).to(device)
self.gmm_variance_value = 0.2
self.gmm_variances = self.gmm_variance_value*torch.ones_like(self.gmm_means).to(device)

sz = self.source_latent_zs
szt = torch.tensor(sz).to(device)

# self.mixture_distribution = torch.distributions.Categorical(torch.ones(self.number_of_components).to(device))
self.mixture_distribution = torch.distributions.Categorical(torch.ones(self.gmm_means.shape[0]).to(device))
self.component_distribution = torch.distributions.Independent(torch.distributions.Normal(self.gmm_means,self.gmm_variances),1)

self.GMM = torch.distributions.MixtureSameFamily(self.mixture_distribution, self.component_distribution)
c = self.GMM.log_prob(szt).detach().cpu().numpy()
ez, _ = self.get_transform(sz)
plt.scatter(ez[:,0],ez[:,1],c=c)
plt.colorbar()
plt.savefig("LP_Colored_Embed.png")
plt.close()

# Reverse GMM
self.reverse_component_distribution = torch.distributions.Independent(torch.distributions.Normal(szt,self.gmm_variances),1)
self.reverse_GMM = torch.distributions.MixtureSameFamily(self.mixture_distribution, self.reverse_component_distribution)

revc = self.reverse_GMM.log_prob(self.gmm_means).detach().cpu().numpy()
revez, _ = self.get_transform(self.target_latent_zs)
plt.scatter(revez[:,0],revez[:,1],c=revc)
plt.colorbar()
plt.savefig("LP_Colored_Embed_Reverse.png")
plt.close()
