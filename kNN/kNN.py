import numpy as np

class KNNModel():
    def __init__(self, n_neighbors=3, dist_func="euclidean"):
        self.n_neighbors = n_neighbors
        self._train_X = None
        self._train_Y = None
        self._ranges = None
        self._min_vals = None
        self._fitted = False
        if dist_func in ["euclidean"]:
            self.dist_func = dist_func
        else:
            raise ValueError(f"Unsupported: {dist_func}")

    def fit(self, X, Y):
        self._train_X, self._ranges, self._min_vals = self._normalize(X)
        self._train_Y = Y.copy()
        self._fitted = True

    def _normalize(self, X):
        minVals = X.min(0)
        maxVals = X.max(0)
        ranges = maxVals-minVals
        normDataSet = np.zeros(np.shape(X))
        m = X.shape[0]
        normDataSet = X - np.tile(minVals, (m,1))
        normDataSet = normDataSet/np.tile(ranges, (m,1))
        return normDataSet, ranges, minVals

    def classify(self, X):
        if not self._fitted:
            raise RuntimeError

        X = (X - self._min_vals) / self._ranges
        n = X.shape[0]
        m = self._train_X.shape[0]

        if self.dist_func == "euclidean":
            d = np.tile(X, (m, 1, 1)).transpose((1, 0, 2)) - self._train_X
            d = np.sqrt((d ** 2).sum(axis=2))
        
        idxs = d.argsort(axis=1)[:, :self.n_neighbors]

        Y = self._train_Y[idxs]
        preds = np.zeros(n, dtype=Y.dtype)

        for i in range(n):
            y, c = np.unique(Y[i], return_counts=True)
            preds[i] = y[c.argmax()]
            
        return preds

    def score_accuracy(self, X, Y):
        preds = self.classify(X)
        correct = np.sum(preds == Y)
        return correct / X.shape[0]
