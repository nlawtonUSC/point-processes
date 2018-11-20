import numpy as np

class Parameter(object):
	"""
	Attributes
	----------
	value : ?
		The initial value of this parameter.
	history : list 
		A list of the history of the value of
		this parameter. Each value in the history
		can be interpreted as a sample from the
		posterior.
	sigma_prop : float
		The standard deviation of the MCMC
		proposal distribution used in training,
		if the parameter is a float.
	"""

	def __init__(self, value, sigma_prop=0.1):
		self.value = value
		self.history = []
		self.sigma_prop = sigma_prop

	def mcmc_proposal(self):
		"""
		Proposes a new value for this parameter
		by sampling from a proposal distribution.
		"""
		raise NotImplementedException

	def logp(self, value, *args):
		"""
		Evaluates the log-likelihood of the model
		as a function of this parameter up to
		a constant.

		Parameters
		----------
		value : ?
			A proposed value for this parameter
		*args : tuple
			A tuple of other model parameters 
			relevant for computing the log-likelihood.
		"""
		raise NotImplementedException

	def mcmc_update(self, *args, **kwargs):
		"""
		Executes a single Metropolis-Hastings
		update for this parameter.

		Parameters
		----------
		*args : tuple
			A list of other parameters relevant
			for computing the log-Hastings ratio.
		**kwargs : dictionary
			A dictionary containing one key, 'record',
			whose value is a bool indicating whether
			parameters should record updates in
			their histories for this MCMC update.
		"""
		vprop = self.mcmc_proposal()
		log_hastings_ratio = self.logp(vprop, *args) - self.logp(self.value, *args)
		if np.log(np.random.uniform()) < log_hastings_ratio:
			self.value = vprop
		if kwargs['record']:
			self.history.append(self.value)

