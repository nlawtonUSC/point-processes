import numpy as np
from parameter import Parameter
from point_process import PointProcess

class PoissonLambda(Parameter):
	"""
	The expected number of events per unit time.
	"""

	def __init__(self, value):
		Parameter.__init__(self, value)


	def mcmc_proposal(self):
		return np.exp(np.log(self.value) + np.random.normal(scale=self.sigma_prop))


	def logp(self, value, events, t_start, t_end):
		N = len(events)
		dT = t_end - t_start
		return (N+1) * np.log(value) - dT * value # the +1 comes from the log-normal proposal distribution.


class Poisson(PointProcess):

	def __init__(self, lam):
		self.lam = PoissonLambda(lam)


	def sample(self, t_start, t_end):
		t = t_start
		events = {'Event_Date':[]}
		while True:
			tau = -np.log(np.random.uniform()) / self.lam.value #exponential random variable
			if t + tau < t_end:
				t += tau
				events['Event_Date'].append(t)
			else:
				break
		events['Event_Date'] = np.array(events['Event_Date'])
		return events


	def log_likelihood(self, events, t_start, t_end):
		N = len(events)
		dT = t_end - t_start
		lam = self.lam.value
		return N * np.log(lam) - dT * lam


	def train_step(self, events, t_start, t_end, record=True):
		self.lam.mcmc_update(events, t_start, t_end, record=record)

