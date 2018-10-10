from sklearn.metrics import confusion_matrix as get_confusion_matrix
import matplotlib.pyplot as plt
from numpy import arange
from itertools import product

class ConfusionMatrix:

    def __init__(self, y_true, y_predicted, classes, normalize=False):
        self.y_true = y_true
        self.y_predicted = y_predicted
        self.classes = classes
        self.normalize = normalize


    def get(self, normalize=None):

        # Determine normalization
        if normalize is None:
            normalize = self.normalize

        return get_confusion_matrix(self.y_true, self.y_predicted)


    def print(self, normalize=None):

        print(self.get(normalize=normalize))


    def plot(self, title="Confusion Matrix", normalize=None):

        confusion_matrix = self.get(normalize=normalize)

        plt.figure()
        plt.title(title)
        # The blue squares
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.colorbar()
        tick_marks = arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=0)    # Use rotation=45 if spacing is an issue
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.

        for i, j in product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.show()
