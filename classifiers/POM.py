# encoding: utf-8
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import inspect

from mord import mord

class POM(BaseEstimator, ClassifierMixin):

	'''
	POM Proportional Odd Model for Ordinal Regression. This class uses POM 
	implementation by Fabian Pedregosa (https://github.com/fabianp/mord)
	
		POM methods:
			fit                     - Fits a model from training data
			predict                 - Performs label prediction
	
		References:
			[1] P. McCullagh, Regression models for ordinal data,  Journal of
				the Royal Statistical Society. Series B (Methodological), vol. 42,
				no. 2, pp. 109–142, 1980.
			[2] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
				F. Fernández-Navarro and C. Hervás-Martínez
				Ordinal regression methods: survey and experimental study
				IEEE Transactions on Knowledge and Data Engineering, Vol. 28. Issue 1
				2016
				http://dx.doi.org/10.1109/TKDE.2015.2457911

	Model Parameters:
		alpha: float
			Regularization parameter. Zero is no regularization, higher values
			increate the squared l2 regularization.

		base_classifier: bool
			Mord offers 2 implementations for this algorithm, 
			True: implements the ordinal logistic model All-Threshold variant
			False: implements the ordinal logistic model Immediate-Threshold variant

	'''

	# Set parameters values
	def __init__(self, alpha=1., base_classifier = False):
		self.alpha = alpha
		self.base_classifier = base_classifier

	def fit(self, X, y):

		"""
		Fit the model with the training data

		Parameters
		----------

		X: {array-like, sparse matrix}, shape (n_samples, n_features)
			Training patterns array, where n_samples is the number of samples
			and n_features is the number of features

		y: array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		self: object
		"""
		

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		# Fit the model
		if self.base_classifier:
			self.classifier_ = mord.LogisticAT(alpha = self.alpha)
		else:
			self.classifier_ = mord.LogisticIT(alpha = self.alpha)

		self.classifier_.fit(X ,y)
		return self

	def predict(self, X):

		"""
		Performs classification on samples in X

		Parameters
		----------

		X : {array-like, sparse matrix}, shape (n_samples, n_features)

		Returns
		-------

		predicted_y : array, shape (n_samples,)
			Class labels for samples in X.
		"""

		# Check is fit had been called
		check_is_fitted(self, ['classifier_'])
		
		# Input validation
		X = check_array(X)

		predicted_y = self.classifier_.predict(X)

		return predicted_y
