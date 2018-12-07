import numpy as np
import matplotlib.pyplot as plt
from poisson import Poisson

t_start = 0.0
t_end = 10.0
lambda_true = 1.0
model_true = Poisson(lambda_true)
events = model_true.sample(t_start, t_end)

lam_init = -np.log(np.random.uniform()) / 0.1
model_infer = Poisson(lam_init)
model_infer.fit(events, t_start, t_end)

lambda_avg = np.mean(model_infer.lam.history)
lambda_var = np.var(model_infer.lam.history)
N = len(events)
dT = t_end - t_start
print 'true posterior mean, var: %f, %f' % ((N+1)/dT, (N+1)/np.square(dT))
print 'inferred posterior mean, var: %f, %f' % (lambda_avg, lambda_var)

figure = plt.figure()
plt.hist(model_infer.lam.history, bins=np.linspace(0.0, 3.0, 50), density=True)
plt.savefig('results/poisson_lambda.png')
