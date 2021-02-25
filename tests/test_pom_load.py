from sys import path as syspath
from os import path as ospath
import ntpath
from shutil import rmtree
import gc

import unittest

import numpy as np
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing

syspath.append('..')
syspath.append(ospath.join('..', 'classifiers'))

from utilities import Utilities


class TestPomLoad(unittest.TestCase):
	"""
	Class testing POM's functionality.

	This classifier is built in classifiers/POM.py.
	"""

	# Getting path to datasets folder
	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_pom_load_dataset")

	# Parameters values for experiments
	values = [-1, -0.33, 0, 0.33, 1, 5 ]
	valuesBase = ["true", "false"]
	
	# Declaring a simple configuration
	general_conf = {"basedir": dataset_path,
					"datasets": ["automobile", "balance-scale", "bondrate", "car", "contact-lenses", "ERA", "ESL", "eucalyptus", "LEV", "newthyroid",
								 "pasture", "squash-stored", "squash-unstored", "SWD", "tae", "toy", "winequality-red"],
					"input_preprocessing": "std",
					"hyperparam_cv_nfolds": 3,
					"jobs": 10,
					"output_folder": "my_runs/",
					"metrics": ["ccr", "mae", "amae", "mze"],
					"cv_metric": "mae"}

	configurations = {
		"pom": {

			"classifier": "POM",
			"parameters": {
				"alpha": values,
				"base_classifier": valuesBase
			}

		}
	}
		
	def test_pom_load(self):
		gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)
		
		print("\n")
		print("###############################")
		print("POM load test")
		print("###############################")

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf, self.configurations, verbose=True)
		util.run_experiment()
		# Saving results information
		util.write_report()

		#Delete all the test results after load test
		rmtree("my_runs")


if __name__ == '__main__':
	unittest.main()