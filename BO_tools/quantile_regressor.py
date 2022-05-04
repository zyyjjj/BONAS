import numpy as np
import torch
from sklearn.linear_model import QuantileRegressor

class QuantileRegressor():
    
    def __init__(self, dataset, quantiles = [0.05, 0.5, 0.95], ifTransformSigmoid=True):
        """Initialize Optimizer object
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        self.__quantiles = quantiles
        self.ifTransformSigmoid = ifTransformSigmoid
    
    def train(self):

        train_X = self.__dataset[0]
        train_X = train_X.cpu().data.numpy()
        train_Y = self.__dataset[1]
        if self.ifTransformSigmoid: # do logit transformation on train_Y
            train_Y = np.log(train_Y / (1 - train_Y))

        qr_models = {}
        for quantile in self.__quantiles:
            qr_model = QuantileRegressor(quantile=quantile, alpha=0).fit(train_X, train_Y)
            qr_models[quantile] = qr_model
        
        self.__qr_models = qr_models

    def predict(self, test_X):
        test_X = test_X.cpu().data.numpy()

        # dictionary with quantile as key and predictions as values
        test_pred_vals = {}
        for quantile in self.__quantiles:
            res = self.__qr_models[quantile].predict(test_X)
            test_pred_vals[quantile] = np.reshape(res, (test_X.shape[0], 1))
        
        if self.ifTransformSigmoid: # inverse logit transform to original values
            test_pred_median_true = 1 / (1 + np.exp(-test_pred_vals[0.5]))
        else:
            test_pred_median_true = test_pred_vals[0.5]

        # TODO: use unit tests to check correctness of the reshaping option
        # TODO: check that the regression_type is correctly "wired" in other parts of the pipeline
        
        # we want it to return predictions for lower, median, upper quantile;
        # transformed predicted median; interquartile range
        return test_pred_vals[0.5], test_pred_vals[0.05], test_pred_vals[0.95], test_pred_median_true, \
            test_pred_vals[0.95] - test_pred_vals[0.05]