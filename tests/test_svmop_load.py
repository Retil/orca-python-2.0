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


class TestSvmopLoad(unittest.TestCase):
	"""
	Class testing SVMOP's functionality.

	This classifier is built in classifiers/SVMOP.py.
	"""

	# Getting path to datasets folder
	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_svmop_load_dataset")

	# Parameters values for experiments
	values = np.logspace(-3, 3, 7).tolist()
	
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
		"svmop_radial": {

			"classifier": "SVMOP",
			"parameters": {
				"C": values,
				"gamma": values
			}

		}
	}
		
	def test_svmop_load(self):
		gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)
		
		print("\n")
		print("###############################")
		print("SVMOP load test")
		print("###############################")

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf, self.configurations, verbose=True)
		util.run_experiment()
		# Saving results information
		util.write_report()

		#Delete all the test results after load test
		rmtree("my_runs")

	def test_svmop_file_exist(self):

		print("\n")
		print("###############################")
		print("SVMOP file exist test")
		print("###############################")

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf, self.configurations, verbose=True)
		util.run_experiment()
		# Saving results information
		util.write_report()
		
		report_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "my_runs")
		
		if not ospath.isdir(report_path):
			raise AssertionError("File does not exist: %s" % str(report_path))

		#Delete all the test results after load test
		rmtree("my_runs")


if __name__ == '__main__':
	unittest.main()