##############################################################################
# Generic dataset handler
##############################################################################

from sklearn.preprocessing import OneHotEncoder as OHE

class dSet(object):
    # Constructor
    def __init__(self, data, labels):
        self._data = data
        self._labels = labels
        self._label_encoder = OHE(sparse=False)
        self._one_hot = self._label_encoder.fit_transform(self._labels)

    # Getters
    @property
    def images(self):
        return self._data
    
    @property
    def labels(self):
        return self._labels

    @property
    def one(self):
        return self._one_hot
 
    # Setter
    def set_one(self, one):
        self._one_hot = one

class Dataset(object):
    
    # Constructor 
    def __init__(self, train_data, test_data, train_labels, test_labels):
        self._train = dSet(train_data, train_labels)
        self._test = dSet(test_data, test_labels)

    # Getters
    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

 
    # Setters
    def set_train_ones(self, one):
        self._train.set_one(one)

    def set_test_ones(self, one):
        self._test.set_one(one)



        
