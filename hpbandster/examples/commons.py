"""
Worker for Examples 1-4
=======================

This class implements a very simple worker used in the firt examples.
"""

import numpy as np
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker


class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """


        #res = numpy.clip(config['x'] + numpy.random.randn()/budget, config['x']/2, 1.5*config['x'])
        res1 = branin([config['x1'], config['x2']])
        res2 = currin_exp([config['x1'], config['x2']])
        time.sleep(self.sleep_interval)

        #print(res1, res2, budget)
        return({
                    'loss': [res1, res2],  # this is the a mandatory field to run hyperband
                    'info': None  # can be used for any user-defined information - also mandatory
                })
    
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x1', lower=-5, upper=10))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x2', lower=0, upper=15))
        return(config_space)

def branin_with_params(x, a, b, c, r, s, t):
	""" Computes the Branin function. """
	x1 = x[0]
	x2 = x[1]
	neg_ret = float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)
	return float(neg_ret)

def branin(x):
	""" Branin function."""
	a = 1
	b = 5.1/(4*np.pi**2)
	c = 5/np.pi
	r = 6
	s = 10
	t = 1/(8*np.pi)
	return branin_with_params(x, a, b, c, r, s, t)

def currin_exp_01(x):
  """ Currin exponential function. """
  x1 = x[0]
  x2 = x[1]
  val_1 = 1 - np.exp(-1/(2 * x2))
  val_2 = (2300*x1**3 + 1900*x1**2 + 2092*x1 + 60) / (100*x1**3 + 500*x1**2 + 4*x1 + 20)
  return float(val_1 * val_2)


def currin_exp(x):
  """ Currint exponential in branin bounds. """
  return -currin_exp_01([x[0] * 15 - 5, x[1] * 15])