from sys import path as syspath
from os import path as ospath

import unittest

import numpy as np
import numpy.testing as npt

syspath.append(ospath.join('..', 'classifiers'))

from POM import POM


class TestPom(unittest.TestCase):
	"""
	Class testing POM's functionality.

	This classifier is built in classifiers/POM.py.
	"""

	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_pom_dataset")

	train_file = np.loadtxt(ospath.join(dataset_path,"train.0"))
	test_file = np.loadtxt(ospath.join(dataset_path,"test.0"))
	'''
	def test_pom_fit_correct(self):
		#Check if this algorithm can correctly classify a toy problem.
		
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		X_test = self.test_file[:,0:(-1)]

		expected_predictions = [ospath.join(self.dataset_path,"expectedPredictions.0")]

		classifiers = [POM(alpha=0.33, base_classifier = False)]

		#Test execution and verification
		for expected_prediction, classifier in zip(expected_predictions, classifiers):
			classifier.fit(X_train, y_train)
			predictions = classifier.predict(X_test)
			expected_prediction = np.loadtxt(expected_prediction)
			npt.assert_array_almost_equal(x= predictions,y= expected_prediction, decimal=1,err_msg="The prediction doesnt match with the desired values")
	'''
	def test_pom_fit_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		X_train_broken = self.train_file[0:(-1),0:(-2)]
		y_train_broken = self.train_file[0:(-1),(-1)]

		#Test execution and verification
		classifier = POM(alpha=1, base_classifier = False)
		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, y_train_broken)
				self.assertIsNone(model, "The POM fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit([], y_train)
				self.assertIsNone(model, "The POM fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, [])
				self.assertIsNone(model, "The POM fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train_broken, y_train)
				self.assertIsNone(model, "The POM fit method doesnt return Null on error")

	def test_pom_predict_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifier = POM(alpha=1, base_classifier = False)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaises(ValueError):
			classifier.predict([])

if __name__ == '__main__':
	unittest.main()
