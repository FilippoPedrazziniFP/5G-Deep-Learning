from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Plot(object):
	"""docstring for Plot"""
	def __init__(self, arg):
		super(Plot, self).__init__()
		self.arg = arg

	@staticmethod
	def visualizing_learning(train_a, train_c, test_a, test_c, y_pred, y_true):
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
	    
	    y_test = np.argmax(y_test, axis=1)
	    
	    print("Confusion Matrix")
	    print(Plot.custom_confusion_matrix(y_test, y_pred, ['Attack', 'Normal']))
	    print("Classification Report")
	    print(classification_report(y_test, y_pred))
	    
	    return

	@staticmethod
	def custom_confusion_matrix(y_true, y_pred, labels=["False", "True"]):
	    cm = confusion_matrix(y_true, y_pred)
	    pred_labels = ["Predicted " + l for l in labels]
	    df = pd.DataFrame(cm, index=labels, columns=pred_labels)
	    return df
