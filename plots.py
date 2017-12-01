from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

class Plot(object):
	"""docstring for Plot"""
	def __init__(self, arg):
		super(Plot, self).__init__()
		self.arg = arg

	@staticmethod
	def visualizing_learning(train_a, train_c, test_a, test_c, plot):
	    if plot == True:
		    plt.title("Accuracy")
		    plt.plot(train_a, label="train")
		    plt.plot(test_a, label="test")
		    plt.grid(True)
		    plt.legend(loc="best")
		    plt.show()
		    
		    plt.title("Loss")
		    plt.plot(train_c, label="train")
		    plt.plot(test_c, label="test")
		    plt.grid(True)
		    plt.legend(loc="best")
		    plt.show()
	
	@staticmethod   
	def report(y_pred, y_test, plot):

		y_test = np.argmax(y_test, axis=1)
		y_pred = np.asarray(y_pred).reshape(-1, 2)
		y_pred = np.argmax(y_pred, axis=1)

		f1_value = f1_score(y_test, y_pred, average=None)
		if plot == True:
			print("Confusion Matrix")
			print(Plot.custom_confusion_matrix(y_test, y_pred, ['Attack', 'Normal']))
			print("Classification Report")
			print(classification_report(y_test, y_pred))
		return f1_value

	@staticmethod
	def custom_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
	    cm = confusion_matrix(y_true, y_pred)
	    pred_labels = ["Predicted " + l for l in labels]
	    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
	    return df

	@staticmethod
	def saving_scores(flags, test_accuracy, f1_value):
		file = open("results/scores.txt", "a")
		file.write("Params: %s --> Accuracy: %s, F1 Score: %s \n" %(flags, test_accuracy, f1_value))
		file.close()
		# print("Saved test score and parameters in a file.")
		return
