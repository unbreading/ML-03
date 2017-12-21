import numpy as np
import ensemble
from sklearn import tree

labels = np.load('labels.npy')
features = np.load('features.npy')

train_size = 800
X = features[:train_size]
y = labels[:train_size]

rng_state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(y)

ada = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier, 12)


ada.fit(X, y)
