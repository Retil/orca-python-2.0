from sys import path as syspath
from os import path as ospath
import ntpath
from shutil import rmtree
import gc
import csv
import unittest
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

syspath.append('..')
syspath.append(ospath.join('..', 'classifiers'))

from utilities import Utilities


class TestPomCompare(unittest.TestCase):
	"""
	Class testing POM's functionality.

	This classifier is built in classifiers/POM.py.
	"""

	# Getting path to datasets folder
	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_compare_dataset")

	# Parameters values for experiments
	values = [-1, -0.33, 0, 0.33, 1, 5 ]
	valuesBase = [True, False]
	
	# Declaring a simple configuration for mze metric
	general_conf_mze = {"basedir": dataset_path,
					"datasets": ['housing', 'abalone', 'calhousing-5', 'census2-5', "contact-lenses", "tae", 
								"balance-scale", "car", "winequality-red", "ERA"],
					"input_preprocessing": "std",
					"hyperparam_cv_nfolds": 5,
					"jobs": -1,
					"output_folder": "my_runs/",
					"metrics": ["mae", "mze"],
					"cv_metric": "mze"}

	# Declaring a simple configuration for mae metric
	general_conf_mae = {"basedir": dataset_path,
					"datasets": ['housing', 'abalone', 'calhousing-5', 'census2-5', "contact-lenses", "tae", 
								"balance-scale", "car", "winequality-red", "ERA"],
					"input_preprocessing": "std",
					"hyperparam_cv_nfolds": 5,
					"jobs": -1,
					"output_folder": "my_runs/",
					"metrics": ["mae", "mze"],
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
		
	def test_pom_compare(self):
		gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)
		
		print("\n")
		print("###############################")
		print("   POM load test MZE metric")
		print("###############################")

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf_mze, self.configurations, verbose=True)
		util.run_experiment()
		# Saving results information
		util.write_report()

		testPath = list(Path(".").rglob("test_summary.csv"))

		testList = []

		with open(testPath[0], mode='r') as csv_file :
			testList = [{key: valor for key, valor in row.items()}
						for row in csv.DictReader(csv_file)]

		names = [dic.get("") for dic in testList]
		mae_mze = [float(dic.get("mae_mean")) for dic in testList]
		mze_mze = [float(dic.get("mze_mean")) for dic in testList]

		#Delete all the test results after load test
		rmtree("my_runs")

		print("\n")
		print("###############################")
		print("   POM load test MAE metric")
		print("###############################")

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf_mae, self.configurations, verbose=True)
		util.run_experiment()
		# Saving results information
		util.write_report()

		testPath = list(Path(".").rglob("test_summary.csv"))

		testList = []

		with open(testPath[0], mode='r') as csv_file :
			testList = [{key: valor for key, valor in row.items()}
						for row in csv.DictReader(csv_file)]

		names = [dic.get("") for dic in testList]
		mae_mae = [float(dic.get("mae_mean")) for dic in testList]
		mze_mae = [float(dic.get("mze_mean")) for dic in testList]

		#Delete all the test results after load test
		rmtree("my_runs")

		#Get the best results from both metrics 
		mae = []
		mze = []

		for i in range(len(mae_mae)):
			if mae_mae[i] < mae_mze[i] :
				mae.append(mae_mae[i])
			else:
				mae.append(mze_mae[i])
		
		for i in range(len(mze_mae)):
			if mze_mae[i] < mze_mze[i] :
				mze.append(mze_mae[i])
			else:
				mze.append(mze_mae[i])
		

		#Extract Data from ORCA Original
		RealnPath = list(Path(".").rglob("ORCA-Results-POM-Real.csv"))
		RegresionPath = list(Path(".").rglob("ORCA-Results-POM-Regresion.csv"))

		#Initialize arrays
		dataObtainReal = np.array([[0.0,0.0,0.0]]*6)
		dataObtainRegresion = np.array([[0.0,0.0,0.0]]*4)

		#Read csv file and retrieve data from Real datasets
		with open(RealnPath[0], mode='r') as csv_file:
			csvReader = csv.reader(csv_file, delimiter="," , quotechar='"')
			
			datasets = next(csvReader)
			datasets = [value for value in datasets if value != '']
			_ = next(csvReader)
			
			for row in csvReader:
				
				#Sum data
				for i in range(len(datasets)):
					dataObtainReal[i, 0] += float(row[ ((i+1)*3)-3 ])
					dataObtainReal[i, 1] += float(row[ ((i+1)*3)-2 ])
					dataObtainReal[i, 2] += float(row[ ((i+1)*3)-1 ])
			
			#Take the mean of the data 
			dataObtainReal[:, :] = dataObtainReal[:, :]/30

		#Read csv file and retrieve data from Regresion datasets
		with open(RegresionPath[0], mode='r') as csv_file:
			csvReader = csv.reader(csv_file, delimiter="," , quotechar='"')
			
			datasets = next(csvReader)
			datasets = [value for value in datasets if value != '']
			_ = next(csvReader)
			
			for row in csvReader:

				#Sum aata
				for i in range(len(datasets)):
					dataObtainRegresion[i, 0] += float(row[ ((i+1)*3)-3 ])
					dataObtainRegresion[i, 1] += float(row[ ((i+1)*3)-2 ])
					dataObtainRegresion[i, 2] += float(row[ ((i+1)*3)-1 ])

			#Take the mean of the data 
			dataObtainRegresion[:, :] = dataObtainRegresion[:, :]/20

		#Sort the mean array in the required format 
		mzeOrca = [dataObtainRegresion[1, 0], dataObtainReal[2, 0], dataObtainRegresion[2, 0], dataObtainReal[3, 0], dataObtainRegresion[3, 0],
					dataObtainReal[0, 0], dataObtainReal[5, 0], dataObtainRegresion[0, 0], dataObtainReal[1,0], dataObtainReal[4,0]]
		maeOrca = [dataObtainRegresion[1, 1], dataObtainReal[2, 1], dataObtainRegresion[2, 1], dataObtainReal[3, 1], dataObtainRegresion[3, 1],
					dataObtainReal[0, 1], dataObtainReal[5, 1], dataObtainRegresion[0, 1], dataObtainReal[1,1], dataObtainReal[4,1]]

		#print results
		print("\n")
		print("###############################")
		print("   POM results")
		print("###############################")

		print('mae Python|ORCA:')
		[print('{0} : {1} \t {2}'.format(names[i], mae[i], maeOrca[i])) for i in range(len(names))]
		print('--------------------------')

		print('mze Python\ORCA:')
		[print('{0} : {1} \t {2}'.format(names[i], mze[i], mzeOrca[i])) for i in range(len(names))]
		print('--------------------------')


		#Make the graphs of the information obtained 
		fig = px.line(x=names, y=maeOrca, color=px.Constant("MAE ORCA"),
             labels=dict(x="DatasetName", y="MAE Mean Value", color="Metric"))
		fig.add_bar(x=names, y=mae, name="MAE ORCA-Python")
		fig.show()

		fig = px.line(x=names, y=mzeOrca, color=px.Constant("MZE ORCA"),
             labels=dict(x="DatasetName", y="MZE Mean Value", color="Metric"))
		fig.add_bar(x=names, y=mze, name="MZE ORCA-Python")
		fig.show()


if __name__ == '__main__':
	unittest.main()