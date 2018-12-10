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
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues) # pylint: disable=E1101
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


    def save(self, title="Confusion Matrix", path=None, normalize=None):

        confusion_matrix = self.get(normalize=normalize)

        plt.figure()
        plt.title(title)
        # The blue squares
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues) # pylint: disable=E1101
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

        if not path[-1] == '/':
            path = path + '/'

        filename = path + title + '.jpg'
        plt.savefig(filename)


class TrainingHistory:

    def __init__(self, historyobject):
        self.hobj = historyobject

    def plot(self):
        plt.subplot(121)
        plt.plot(self.hobj.history['acc'], 'r-', lw=0.5)
        plt.plot(self.hobj.history['val_acc'], 'b-', lw=0.5)
        plt.title('Train vs Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(top=1, bottom=0)
        plt.legend(['Train Accuracy', 'Test Accuracy'], loc='best')

        plt.subplot(122)
        plt.plot(self.hobj.history['loss'], 'r-', lw=0.5)
        plt.plot(self.hobj.history['val_loss'], 'b-', lw=0.5)
        plt.title('Train vs Test Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim(top=1, bottom=0)
        plt.legend(['Train Loss', 'Test Loss'], loc='best')
        plt.show()

    def save(self, path):

        plt.subplot(121)
        plt.plot(self.hobj.history['acc'], 'r-', lw=0.5)
        plt.plot(self.hobj.history['val_acc'], 'b-', lw=0.5)
        plt.title('Train vs Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(top=1, bottom=0)
        plt.legend(['Train Accuracy', 'Test Accuracy'], loc='best')

        plt.subplot(122)
        plt.plot(self.hobj.history['loss'], 'r-', lw=0.5)
        plt.plot(self.hobj.history['val_loss'], 'b-', lw=0.5)
        plt.title('Train vs Test Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim(top=1, bottom=0)
        plt.legend(['Train Loss', 'Test Loss'], loc='best')
        plt.savefig(path + 'Training History.jpg')