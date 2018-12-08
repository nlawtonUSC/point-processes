import numpy as np
import matplotlib.pyplot as plt
from hawkes import Hawkes

# generate synthetic data
t_start = 0.0
t_end = 50.0
mu_true = 0.5
alpha_true = 0.9
beta_true = 1.0
model_true = Hawkes(mu_true, alpha_true, beta_true)
events = model_true.sample(t_start, t_end)
event_times = events['Event_Date']
N = len(event_times)
print 'number of events: ', N

# train model
mu_init = -np.log(np.random.uniform()) / 0.1
alpha_init = -np.log(np.random.uniform()) / 0.1
beta_init = -np.log(np.random.uniform()) / 0.1
model_infer = Hawkes(mu_init, alpha_init, beta_init)
model_infer.fit(events, t_start, t_end, burn_in=5000, train_its=20000)

# compute approximate posterior statistics
mu_avg = np.mean(model_infer.mu.history)
mu_std = np.std(model_infer.mu.history)
alpha_avg = np.mean(model_infer.alpha.history)
alpha_std = np.std(model_infer.alpha.history)
beta_avg = np.mean(model_infer.beta.history)
beta_std = np.std(model_infer.beta.history)

parents_matrix = np.zeros((N+1, N+1))
for event_idx in range(N):
	parent_history = model_infer.parent_params[event_idx].history
	parent_histogram = np.histogram(parent_history, bins=event_idx+1, range=(-1, event_idx-1), density=True)[0]
	parents_matrix[event_idx,:event_idx+1] = parent_histogram

print 'mu avg/std: %f, %f' % (mu_avg, mu_std)
print 'alpha avg/std: %f, %f' % (alpha_avg, alpha_std)
print 'beta avg/std: %f, %f' % (beta_avg, beta_std)

# display visualizations
fig = plt.figure()
plt.xlim((t_start, t_end))
plt.plot(event_times, np.zeros_like(event_times), 'o', color='r', fillstyle='none')
plt.savefig('results/hawkes_events.png')

fig = plt.figure(figsize=(24,24))
plt.imshow(parents_matrix, interpolation='nearest', vmin=0, vmax=1)
plt.axis('off')
plt.savefig('results/hawkes_parents.png')

figure = plt.figure()
plt.hist(model_infer.mu.history, bins=np.linspace(mu_avg - 3 * mu_std, mu_avg + 3 * mu_std, 50), density=True)
plt.savefig('results/hawkes_mu.png')

figure = plt.figure()
plt.hist(model_infer.alpha.history, bins=np.linspace(alpha_avg - 3 * alpha_std, alpha_avg + 3 * alpha_std, 50), density=True)
plt.savefig('results/hawkes_alpha.png')

figure = plt.figure()
plt.hist(model_infer.beta.history, bins=np.linspace(beta_avg - 3 * beta_std, beta_avg + 3 * beta_std, 50), density=True)
plt.savefig('results/hawkes_beta.png')

dt = 0.01
xt = np.linspace(t_start, t_end + 10.0, (t_end + 10.0)/dt)
inferred_rate = np.zeros_like(xt)
inferred_rate += model_infer.exp_decay(xt, mu_avg, 0.0, 0.0)
for event_time in event_times:
	idx = (event_time/dt).astype(np.int32)
	inferred_rate[idx:] += model_infer.exp_decay(xt[idx:] - event_time, 0.0, alpha_avg, beta_avg)
fig = plt.figure()
plt.plot(xt, inferred_rate)
plt.savefig('results/hawkes_inferred_rate.png')





