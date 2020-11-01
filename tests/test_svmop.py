from sys import path as syspath
from os import path as ospath

import unittest

import numpy as np
import numpy.testing as npt

syspath.append(ospath.join('..', 'classifiers'))

from SVMOP import SVMOP


class TestSvmop(unittest.TestCase):
	"""
	Class testing SVMOP's functionality.

	This classifier is built in classifiers/SVMOP.py.
	"""

	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_svmop_dataset")
	
	train_file = np.loadtxt(ospath.join(dataset_path,"train.0"))
	test_file = np.loadtxt(ospath.join(dataset_path,"test.0"))

	def test_svmop_fit_correct(self):
		#Check if this algorithm can correctly classify a toy problem.
		
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		X_test = self.test_file[:,0:(-1)]

		expected_predictions = [ospath.join(self.dataset_path,"expectedPredictions.0"), 
								ospath.join(self.dataset_path,"expectedPredictions.1"),
								ospath.join(self.dataset_path,"expectedPredictions.2")
								]

		classifiers = [SVMOP(g=0.1, c=0.1),
					SVMOP(g=1, c=1),
					SVMOP(g=10, c=10)
					]

		#Test execution and verification
		for expected_prediction, classifier in zip(expected_predictions, classifiers):
			classifier.fit(X_train, y_train)
			predictions = classifier.predict(X_test)
			expected_prediction = np.loadtxt(expected_prediction)
			npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")

	def test_svmop_fit_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		X_train_broken = self.train_file[:(-1),0:(-1)]
		y_train_broken = self.train_file[0:(-1),(-1)]

		#Test execution and verification
		classifier = SVMOP(g=0.1, c=1)
		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, y_train_broken)
				self.assertIsNone(model, "The SVMOP fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit([], y_train)
				self.assertIsNone(model, "The SVMOP fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, [])
				self.assertIsNone(model, "The SVMOP fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train_broken, y_train)
				self.assertIsNone(model, "The SVMOP fit method doesnt return Null on error")

	def test_svmop_predict_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifier = SVMOP(g=0.1, c=1)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaises(ValueError):
			classifier.predict([])


if __name__ == '__main__':
	unittest.main()
