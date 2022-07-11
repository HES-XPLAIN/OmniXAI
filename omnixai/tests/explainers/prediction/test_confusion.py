import unittest
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from omnixai.explainers.prediction import PredictionAnalyzer


class TestConfusionMatrix(unittest.TestCase):

    def setUp(self) -> None:
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target

        random_state = np.random.RandomState(0)
        n_samples, n_features = x.shape
        x = np.c_[x, random_state.randn(n_samples, 20 * n_features)]

        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=0.5, random_state=0)
        classifier = svm.SVC(kernel="linear", probability=True, random_state=0)
        classifier.fit(x_train, y_train)

        self.x_test = x_test
        self.y_test = y_test
        self.classifier = classifier

    def test_confusion(self):
        explainer = PredictionAnalyzer(
            mode="classification",
            predict_function=lambda x: self.classifier.predict_proba(x),
            test_data=self.x_test,
            test_targets=self.y_test
        )
        explanations = explainer._confusion_matrix()
        explanations.plotly_plot(class_names=["a", "b", "c"])
        explanations.plot(class_names=["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
