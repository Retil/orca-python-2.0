import scipy
from libsvmWeight.python.svmutil import *
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class SVMOP(BaseEstimator, ClassifierMixin):
	"""
	SVMOP Support vector machines using Frank & Hall method for ordinal
	regression (by binary decomposition). This class uses libsvm-weights
	for SVM training (https://www.csie.ntu.edu.tw/~cjlin/libsvm).
	  SVMOP methods:
		 fitpredict               - runs the corresponding algorithm,
									  fitting the model and testing it in a dataset.
		 fit                        - Fits a model from training data
		 predict                    - Performs label prediction
	
	  References:
		[1] E. Frank and M. Hall, "A simple approach to ordinal classification"
			in Proceedings of the 12th European Conference on Machine Learning,
			ser. EMCL'01. London, UK: Springer-Verlag, 2001, pp. 145–156.
			https://doi.org/10.1007/3-540-44795-4_13
		[2] W. Waegeman and L. Boullart, "An ensemble of weighted support
			vector machines for ordinal regression", International Journal
			of Computer Systems Science and Engineering, vol. 3, no. 1,
			pp. 47–51, 2009.
		[3] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
			F. Fernández-Navarro and C. Hervás-Martínez
			Ordinal regression methods: survey and experimental study
			IEEE Transactions on Knowledge and Data Engineering, Vol. 28. Issue 1
			2016
			http://dx.doi.org/10.1109/TKDE.2015.2457911

	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"

	"""

	# Set parameters values
	def __init__(self, g=1, c=1):

		self.g = g
		self.c = c


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
		self.classes_ = np.unique(y)

		# Set the default g value if necessary
		if self.g == None:
			self.g = 1 / np.size(X, 1)
		
		# Fit the model
		options = "-b 1 -t 2 -c {} -g {} -q".format(str(self.c), str(self.g))
		#self.classifier_ = svm.fit(y.tolist(), X.tolist(), options)


		# Give each train input its corresponding output label
		# for each binary classifier
		coding_matrix = np.triu((-2 * np.ones(len(self.classes_) - 1))) + 1
		coding_matrix = np.vstack([coding_matrix, np.ones((1, len(self.classes_) -1))])

		class_labels = coding_matrix[(np.digitize(y, self.classes_) - 1), :].astype(int)

		self.classifiers_ = []
		# Fitting n_targets - 1 classifiers
		for n in range(len(class_labels[0,:])):

			estimator = svm_train( np.ravel(class_labels[np.where(class_labels[:,n] != 0), n].T) , 
								   X[np.where(class_labels[:,n] != 0)], options)

			self.classifiers_.append(estimator)

		#self.classifier_ = svm_train(y.tolist(), X.tolist(), options)
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
		check_is_fitted(self, ['classifiers_'])

		# Input validation
		X = check_array(X)


		# Getting predicted labels for dataset from each classifier
		'''
		auxarray = np.array(list(map(lambda c: svm_predict([], X, c, '-b 1'), self.classifiers_)), list).T
		print(auxarray)
		predictions = []
		for i in range(len(self.classes_)-1):
			newarray = auxarray[2,i]
			for n in range(len(newarray)):
				untuple1, untuple2 = newarray[n]
				predictions.append([untuple1,untuple2])
		predictions = np.array(predictions)
		'''
		probTs = []
		for n in range(len(self.classifiers_)):
			p_val, p_acc, prediction = svm_predict([], X, self.classifiers_[n], '-b 1')
			#prediction = list(prediction)
			probTs.append(prediction)

		predictions = np.array(probTs).T
		predictions = predictions[0,:]

		# Probability for each pattern of dataset
		predicted_proba_y = np.empty([(predictions.shape[0]), (predictions.shape[1] + 1)])

		# Probabilities of each set to belong to the first ordinal class
		predicted_proba_y[:,0] = 1 - predictions[:,0]

		# Probabilities for the central classes
		predicted_proba_y[:,1:-1] = predictions[:,:-1] - predictions[:,1:]

		# Probabilities of each set to belong to the last class
		predicted_proba_y[:,-1] = predictions[:,-1]

		predicted_y = self.classes_[np.argmax(predicted_proba_y, axis=1)]

		return predicted_y

	def computeWeight(self, p, targets):
		"""
		Compute the weight

			References:
		[1] WAEGEMAN, Willem; BOULLART, Luc. An ensemble of weighted support 
			vector machines for ordinal regression. International Journal of 
			Computer Systems Science and Engineering, 2009, vol. 3, no 1, p. 
			47-51.

		Parameters
		----------

		p : interger
		targets : {array-like, sparse matrix}, shape (n_samples, n_features)

		Returns
		-------

		weight : array, shape (n_samples,)
			Weighted array.
		"""

		summ = sum([i+1 for i in targets if i <= p])
		weight = [i if i > p else (p+1-i)/summ for i in targets]

		summ = sum([i-p for i in targets if i >p])
		weight = [i if i <= p else (i-p)/summ for i in weight]

		return weight
