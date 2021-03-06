import numpy as np
import matplotlib.pyplot as plt
from nonhomogenous_poisson import NonhomogenousPoisson

# generate synthetic data
t_start = 0.0
t_end = 100.0
mu_true = 0.5
alpha_true = 50.0
beta_true = 1.0
model_true = NonhomogenousPoisson(mu_true, alpha_true, beta_true)
events = model_true.sample(t_start, t_end)
print 'total num. events: ', len(events['Event_Date'])

# train model
mu_init = -np.log(np.random.uniform()) / 0.1
alpha_init = -np.log(np.random.uniform()) / 0.1
beta_init = -np.log(np.random.uniform()) / 0.1
model_infer = NonhomogenousPoisson(mu_init, alpha_init, beta_init)
model_infer.fit(events, t_start, t_end, burn_in=5000, train_its=20000)

# compute approximate posterior statistics
mu_avg = np.mean(model_infer.mu.history)
mu_std = np.std(model_infer.mu.history)
alpha_avg = np.mean(model_infer.alpha.history)
alpha_std = np.std(model_infer.alpha.history)
beta_avg = np.mean(model_infer.beta.history)
beta_std = np.std(model_infer.beta.history)

parents_avg = []
for i in range(len(events['Event_Date'])):
	parents_avg.append(np.mean(model_infer.parent_params[i].history))
num_imm = -np.sum(parents_avg)

print 'mu avg/std: %f, %f' % (mu_avg, mu_std)
print 'alpha avg/std: %f, %f' % (alpha_avg, alpha_std)
print 'beta avg/std: %f, %f' % (beta_avg, beta_std)
print 'avg. number of immigrants: ', num_imm
print 'avg. number of offspring: ', len(events['Event_Date']) - num_imm

# display approximate posterior
fig = plt.figure()
plt.xlim((t_start, t_end))
plt.plot(events['Event_Date'], np.zeros_like(events['Event_Date']), 'o', color='r', fillstyle='none')
plt.savefig('results/nonhomogenous_events.png')

figure = plt.figure()
plt.hist(model_infer.mu.history, bins=np.linspace(mu_avg - 3 * mu_std, mu_avg + 3 * mu_std, 50), density=True)
plt.savefig('results/nonhomogenous_mu.png')

figure = plt.figure()
plt.hist(model_infer.alpha.history, bins=np.linspace(alpha_avg - 3 * alpha_std, alpha_avg + 3 * alpha_std, 50), density=True)
plt.savefig('results/nonhomogenous_alpha.png')

figure = plt.figure()
plt.hist(model_infer.beta.history, bins=np.linspace(beta_avg - 3 * beta_std, beta_avg + 3 * beta_std, 50), density=True)
plt.savefig('results/nonhomogenous_beta.png')

