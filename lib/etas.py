import numpy as np
from parameter import Parameter
from point_process import PointProcess
from hawkes import HawkesMu, HawkesAlpha, HawkesBeta, Hawkes
from nonhomogenous_poisson import NonhomogenousPoisson

class EtasParent(Parameter):

	def __init__(self, value, event_idx, loc_var, wx, wy):
		Parameter.__init__(self, value)
		self.event_idx = event_idx
		self.loc_var = loc_var
		self.wx = wx
		self.wy = wy


	def mcmc_proposal(self):
		return np.random.randint(self.event_idx + 1) - 1


	def logp(self, parent, mu, alpha, beta, event_times, event_locs, t_start, t_end):
		if parent == -1:
			return np.log(mu) - np.log(self.wx * self.wy)
		else:
			child_time = event_times[self.event_idx]
			parent_time = event_times[parent]
			dt = child_time - parent_time
			child_loc = event_locs[self.event_idx]
			parent_loc = event_locs[parent]
			return np.log(alpha) + np.log(beta) - beta * dt - (0.5/self.loc_var) * np.sum(np.square(child_loc - parent_loc))


class Etas(PointProcess):

	def __init__(self, mu, alpha, beta, loc_var, wx, wy):
		self.mu = HawkesMu(mu)
		self.alpha = HawkesAlpha(alpha)
		self.beta = HawkesBeta(beta)
		self.loc_var = loc_var
		self.wx = wx
		self.wy = wy


	def exp_decay(self, t, mu, alpha, beta):
		return mu + alpha * beta * np.exp(-beta * t)


	def sample(self, t_start, t_end):
		t = t_start
		event_times = np.array([])
		event_locs = np.zeros((0,2))
		parents = [(0.0, self.mu.value, 0.0, 0.0, 0.0, 0.0)] # (t, mu, alpha, beta, x, y)
		while len(parents) > 0:
			t_p, mu_p, alpha_p, beta_p, x_p, y_p = parents.pop()
			parent_process = NonhomogenousPoisson(mu_p, alpha_p, beta_p)
			child_times = parent_process.sample(t_p, t_end)
			if mu_p == 0.0:
				child_locs = np.array([(np.random.normal(scale=np.sqrt(self.loc_var)) + x_p, np.random.normal(scale=np.sqrt(self.loc_var)) + y_p) for i in range(len(child_times))])
			else:
				child_locs = np.array([(np.random.uniform(0.0, self.wx), np.random.uniform(0.0, self.wy)) for i in range(len(child_times))])
			if len(child_times > 0):
				event_times = np.concatenate([event_times, child_times])
				event_locs = np.concatenate([event_locs, child_locs])
				for c in range(len(child_times)):
					parents.append((child_times[c], 0.0, self.alpha.value, self.beta.value, child_locs[c][0], child_locs[c][1]))
		sort_inds = event_times.argsort()
		return event_times[sort_inds], event_locs[sort_inds]


	def train_step(self, event_times, event_locs, t_start, t_end, record=True):
		self.mu.mcmc_update(self.parents, t_start, t_end, record=record)
		self.alpha.mcmc_update(self.beta.value, self.parents, event_times, t_start, t_end, record=record)
		self.beta.mcmc_update(self.alpha.value, self.parents, event_times, t_start, t_end, record=record)
		for i in range(100):
			child_idx = np.random.randint(len(event_times))
			self.parent_params[child_idx].mcmc_update(self.mu.value, self.alpha.value, self.beta.value, event_times, event_locs, t_start, t_end, record=record)
			self.parents[child_idx] = self.parent_params[child_idx].value


	def fit(self, event_times, event_locs, t_start, t_end, burn_in=5000, train_its=20000):
		self.parents = np.zeros(len(event_times), dtype=np.int32)
		self.parent_params = []
		for event_idx in range(len(event_times)):
			parent_init = -1
			self.parents[event_idx] = parent_init
			self.parent_params.append(EtasParent(parent_init, event_idx, self.loc_var, self.wx, self.wy))
		PointProcess.fit(self, event_times, event_locs, t_start, t_end, burn_in, train_its)


