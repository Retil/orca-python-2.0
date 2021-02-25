import scipy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn import svm

class SVMOP(BaseEstimator, ClassifierMixin):
	"""
	SVMOP Support vector machines using Frank & Hall method for ordinal
	regression (by binary decomposition). This class uses SVC for 
	SVM training 
	(https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

	  SVMOP methods:
		 fit                        - Fits a model from training data
		 predict                    - Performs label prediction
		 computeWeight              - Compute the weight for training

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

	"-gamma gamma : set gamma in kernel function (default 1/num_features)\n"
	"-C cost : Regularization parameter. The strength of the regularization is 
			   inversely proportional to C."

	"""

	# Set parameters values
	def __init__(self, C= 1, gamma = 1 ):
		self.C = C
		self.gamma = gamma


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

		self.X_ = X
		self.y_ = y

		# Store the classes seen during fit
		self.classes_ = np.unique(y)

		# Give each train input its corresponding output label
		# for each binary classifier

		coding_matrix = np.triu((-2 * np.ones(len(self.classes_) - 1))) + 1
		coding_matrix = np.vstack([coding_matrix, np.ones((1, len(self.classes_) -1))])

		class_labels = coding_matrix[(np.digitize(y, self.classes_) - 1), :].astype(int)

		self.classifiers_ = []
		# Fitting n_targets - 1 classifiers
		for n in range(len(class_labels[0,:])):

			weightsTrain = self.computeWeight(n, y)

			estimator = svm.SVC(C = self.C, gamma = self.gamma, probability = True, kernel = 'rbf', )
			estimator.fit(X[np.where(class_labels[:,n] != 0)],
						  np.ravel(class_labels[np.where(class_labels[:,n] != 0), n].T), sample_weight = weightsTrain)

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
		predictions = np.array(list(map(lambda c: c.predict_proba(X)[:,1], self.classifiers_))).T

		predicted_proba_y = self.binaryDecomposition(predictions)

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


	def binaryDecomposition(seld, predictions):
		"""
		Returns the probability for each pattern of dataset to
		belong to each one of the original targets.	Transforms from n-1
		subproblems to the original ordinal problem with n targets.

			References:
		[2] W. Waegeman and L. Boullart, "An ensemble of weighted support
			vector machines for ordinal regression", International Journal
			of Computer Systems Science and Engineering, vol. 3, no. 1,
			pp. 47–51, 2009.

		Parameters
		----------

		predicted: array, shape (n_samples, n_targets-1)

		Returns
		-------

		predicted_proba_y: array, shape (n_samples, n_targets)
			Class labels predicted for samples in dataset X.
		"""

		# Probability for each pattern of dataset
		predicted_proba_y = np.empty([(predictions.shape[0]), (predictions.shape[1] + 1)])

		# Probabilities of each set to belong to the first ordinal class
		predicted_proba_y[:,0] = 1 - predictions[:,0]

		# Probabilities for the central classes
		predicted_proba_y[:,1:-1] = predictions[:,:-1] - predictions[:,1:]

		# Probabilities of each set to belong to the last class
		predicted_proba_y[:,-1] = predictions[:,-1]

		return predicted_proba_y