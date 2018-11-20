class PointProcess(object):

	def __init__(self, *args):
		"""
		Parameters
		----------
		*args : tuple
			A tuple containing intitial values
			for model parameters
		"""
		raise NotImplementedException


	def sample(self, t_start, t_end):
		"""
		Samples a list of event times from the
		current point process within a specified 
		time window.

		Parameters
		----------
		t_start : float
			The starting time of the sampling window.
		t_end: float
			The ending time of the sampling window.

		Returns
		-------
		list
			List of event times
		"""
		raise NotImplementedException


	def log_likelihood(self, event_times):
		"""
		Evaluates the log-likelihood of a given
		list of event times under the current
		model.

		Parameters
		----------
		event_times : list
			A sorted list of observed event times.

		Returns
		-------
		float
			The log-likelihood of the event times under the model.
		"""
		raise NotImplementedException


	def train_step(self, event_times, t_start, t_end, record=True):
		"""
		Executes one step of training. In
		Bayesian statistics, this means sampling
		from a Markov Chain whose stationary
		distribution is the true posterior
		distribution over the parameters given
		the observed event times.

		Parameters
		----------
		event_times : list
			A sorted list of observed event times.
		t_start : float
			The starting time of the observed time window.
		t_end : float
			The ending time of the observed time window.
		record : bool
			Whether to record the parameter updates in
			their respective histories.
		"""
		raise NotImplementedException


	def fit(self, event_times, t_start, t_end, burn_in=5000, train_its=20000):
		"""
		Fit the model to data by executing
		multiple training steps.

		Parameters
		----------
		event_times : list
			A sorted list of observed event times.
		t_start : float
			The starting time of the observed time window.
		t_end : float
			The ending time of the observed time window.
		burn_in : int
			The number of training steps in which
			parameters will not record updates in
			their histories. This reduces
			initialization bias.
		train_its : int
			The number of training steps to execute.
		"""
		for it in range(train_its):
			record = it >= burn_in
			self.train_step(event_times, t_start, t_end, record=record)




