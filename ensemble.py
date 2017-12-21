import pickle
import numpy as np
import os


class AdaBoostClassifier:

    def __init__(self, weak_classifier, n_weakers_limit):

        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier_list = None
        self.classifier_weight_list = None

    def fit(self, X, y):
        sample_size, feature_size = X.shape
        sample_weight = np.ones(shape=[sample_size]) / feature_size
        self.weak_classifier_list = []
        self.classifier_weight_list = []
        for i in range(self.n_weakers_limit):
            weak_classifier = self.weak_classifier(criterion='entropy', max_depth=4)
            weak_classifier.fit(X, y, sample_weight=sample_weight)
            self.weak_classifier_list.append(weak_classifier)
            with open(r"./model/tree_" + str(i) + ".dot", 'wb') as f:
                pickle.dump(weak_classifier, f)
            print("Finish No." + str(i) + " weak classifier.")
            weak_predict = weak_classifier.predict(X)
            error = np.sum(sample_weight * ((weak_predict != y).astype(float)))
            delta = 1e-6
            classifier_weight = np.log(1. / (error + delta) - 1) / 2.
            self.classifier_weight_list.append(classifier_weight)
            sample_weight = sample_weight * np.exp(-classifier_weight * y * weak_predict)
            sample_weight /= np.sum(sample_weight)
        np.save('classifier_weight_list.npy', np.array(self.classifier_weight_list))
        return

    def predict(self, X, threshold=0):
        self.feed_self()

        predicts = []
        for weak_classifier in self.weak_classifier_list:
            predicts.append(weak_classifier.predict(X))
        predicts = np.array(predicts)
        for i in range(0, len(predicts)):
            predicts[i] *= self.classifier_weight_list[i]
        predict = np.sum(predicts, axis=0)
        predict -= threshold
        predict /= np.abs(predict)
        return predict

    def weak_predict(self, X, classifier_number, threshold=0):
        self.feed_self()
        predict = self.weak_classifier_list[classifier_number].predict(X)
        predict -= threshold
        predict /= np.abs(predict)
        return predict


    def feed_self(self):
        if self.weak_classifier_list is None:
            self.weak_classifier_list = []
            for filename in os.listdir(r"./model"):
                if filename.endswith('dot'):
                    with open(r'./model/' + filename, 'rb') as f:
                        self.weak_classifier_list.append(pickle.load(f))
        if self.classifier_weight_list is None:
            self.classifier_weight_list = np.load('classifier_weight_list.npy')