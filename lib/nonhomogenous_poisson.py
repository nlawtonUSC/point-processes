import numpy as np
from parameter import Parameter
from point_process import PointProcess

class NonhomogenousMu(Parameter):

	def __init__(self, value):
		Parameter.__init__(self, value)


	def mcmc_proposal(self):
		return np.exp(np.log(self.value) + np.random.normal(scale=self.sigma_prop))


	def logp(self, mu, parents, t_start, t_end):
		imm_idx = np.where(parents == -1)[0]
		num_imm = imm_idx.size
		dT = t_end - t_start
		return (num_imm + 1) * np.log(mu) - dT * mu # the +1 comes from the log-normal proposal distribution.


class NonhomogenousAlpha(Parameter):

	def __init__(self, value):
		Parameter.__init__(self, value)


	def mcmc_proposal(self):
		return np.exp(np.log(self.value) + np.random.normal(scale=self.sigma_prop))


	def logp(self, alpha, beta, parents, event_times, t_start, t_end):
		offspring_idx = np.where(parents != -1)[0]
		num_offspring = offspring_idx.size
		dT = t_end - t_start
		return (num_offspring + 1) * np.log(alpha) - alpha * (1.0 - np.exp(-beta * dT))


class NonhomogenousBeta(Parameter):

	def __init__(self, value):
		Parameter.__init__(self, value)


	def mcmc_proposal(self):
		return np.exp(np.log(self.value) + np.random.normal(scale=self.sigma_prop))


	def logp(self, beta, alpha, parents, event_times, t_start, t_end):
		offspring_idx = np.where(parents != -1)[0]
		num_offspring = offspring_idx.size
		offspring_times = event_times[offspring_idx]
		dt = offspring_times - t_start
		dT = t_end - t_start
		return (num_offspring + 1) * np.log(beta) - beta * np.sum(dt) - alpha * (1.0 - np.exp(-beta * dT))


class NonhomogenousParent(Parameter):

	def __init__(self, value, event_idx):
		Parameter.__init__(self, value)
		self.event_idx = event_idx


	def mcmc_proposal(self):
		return np.random.randint(2) - 1


	def logp(self, parent, mu, alpha, beta, event_times, t_start, t_end):
		if parent == -1:
			return np.log(mu)
		else:
			child_time = event_times[self.event_idx]
			dt = child_time - t_start
			return np.log(alpha) + np.log(beta) - beta * dt


class NonhomogenousPoisson(PointProcess):

	def __init__(self, mu, alpha, beta):
		self.mu = NonhomogenousMu(mu)
		self.alpha = NonhomogenousAlpha(alpha)
		self.beta = NonhomogenousBeta(beta)


	def exp_decay(self, t, mu, alpha, beta):
		return mu + alpha * beta * np.exp(-beta * t)


	def sample(self, t_start, t_end):
		t = t_start
		event_times = []
		while True:
			lam_prev = self.exp_decay(t-t_start, self.mu.value, self.alpha.value, self.beta.value) + 1e-20 # for numerical stability
			t += -np.log(np.random.uniform()) / lam_prev
			lam_next = self.exp_decay(t-t_start, self.mu.value, self.alpha.value, self.beta.value)
			prob_accept = lam_next / lam_prev
			if t > t_end:
				break
			elif np.random.uniform() < prob_accept:
				event_times.append(t)
		return np.array(event_times)


	def train_step(self, event_times, t_start, t_end, record=True):
		self.mu.mcmc_update(self.parents, t_start, t_end, record=record)
		self.alpha.mcmc_update(self.beta.value, self.parents, event_times, t_start, t_end, record=record)
		self.beta.mcmc_update(self.alpha.value, self.parents, event_times, t_start, t_end, record=record)
		for i in range(10):
			child_idx = np.random.randint(len(event_times))
			self.parent_params[child_idx].mcmc_update(self.mu.value, self.alpha.value, self.beta.value, event_times, t_start, t_end, record=record)
			self.parents[child_idx] = self.parent_params[child_idx].value


	def fit(self, event_times, t_start, t_end, burn_in=5000, train_its=20000):
		self.parents = np.zeros(len(event_times), dtype=np.int32)
		self.parent_params = []
		for event_idx in range(len(event_times)):
			parent_init = np.random.randint(2) - 1
			self.parents[event_idx] = parent_init
			self.parent_params.append(NonhomogenousParent(parent_init, event_idx))
		PointProcess.fit(self, event_times, t_start, t_end, burn_in, train_its)


