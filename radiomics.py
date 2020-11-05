import random
import os

import pandas as pd
from pandas import read_csv

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

import radiomics

class Radiomics:
    """This class encapsulates the Friedman1 test for a regressor
    """

    DATASET_URL = os.path.join(os.getcwd(), 'dataset', 'dataset.csv')
    LABELS_URL = os.path.join(os.getcwd(), 'dataset', 'radiogenomics_labels.csv')
    #'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5
    
    def __init__(self, randomSeed):
        """
        :param randomSeed: random seed value used for reproducible results
        """
        self.randomSeed = randomSeed
        
        # read the dataset
        self.data = read_csv(self.DATASET_URL)
        self.radiogenomic_labels = read_csv(self.LABELS_URL)
        
        self.data_preprocessing()

        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

        self.classifier = AdaBoostClassifier(random_state=self.randomSeed)

    def __len__(self):
        """
        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def getMeanAccuracy(self, zeroOneList):
        """
        returns the mean accuracy measure of the calssifier, calculated using k-fold validation process,
        using the features selected by the zeroOneList
        :param zeroOneList: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.
        :return: the mean accuracy measure of the calssifier when using the features selected by the zeroOneList
        """

        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)

        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold, scoring='accuracy')

        # return mean accuracy:
        return cv_results.mean()

    def data_preprocessing(self):
        self.data['Case ID'] = None

        for i, image in enumerate(self.data['Image']):
            self.data.loc[i, 'Case ID'] = image.split('.')[0]

        self.data = pd.merge(self.data, self.radiogenomic_labels[['Case ID', 'Survival Status']], left_on='Case ID', right_on='Case ID', how='left')

        self.data.drop(['Mask', 'Image', 'Case ID'], axis=1, inplace=True)
        self.data.dropna(inplace=True)

        # separate to input features and resulting category (last column):
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        self.y.replace({'Alive': 1, 'Dead': 0}, inplace=True)

        scaler = MinMaxScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X))

        self.X.columns = self.X.columns
        
class Ada_Boost_Radiomics:
    """This class encapsulates the Friedman1 test for a regressor
    """

    DATASET_URL = os.path.join(os.getcwd(), 'dataset', 'dataset.csv')
    LABELS_URL = os.path.join(os.getcwd(), 'dataset', 'radiogenomics_labels.csv')
    #'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5
    
    def __init__(self, randomSeed):
        """
        :param randomSeed: random seed value used for reproducible results
        """
        self.randomSeed = randomSeed
        
        # read the dataset
        self.data = read_csv(self.DATASET_URL)
        self.radiogenomic_labels = read_csv(self.LABELS_URL)
        
        self.data_preprocessing()

        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

        self.classifier = AdaBoostClassifier(random_state=self.randomSeed)

    def __len__(self):
        """
        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def getMeanAccuracy(self, zeroOneList):
        """
        returns the mean accuracy measure of the calssifier, calculated using k-fold validation process,
        using the features selected by the zeroOneList
        :param zeroOneList: a list of binary values corresponding the features in the dataset. A value of '1'
        represents selecting the corresponding feature, while a value of '0' means that the feature is dropped.
        :return: the mean accuracy measure of the calssifier when using the features selected by the zeroOneList
        """

        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)

        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, currentX, self.y, cv=self.kfold, scoring='accuracy')

        # return mean accuracy:
        return cv_results.mean()

    def data_preprocessing(self):
        self.data['Case ID'] = None

        for i, image in enumerate(self.data['Image']):
            self.data.loc[i, 'Case ID'] = image.split('.')[0]

        self.data = pd.merge(self.data, self.radiogenomic_labels[['Case ID', 'Survival Status']], left_on='Case ID', right_on='Case ID', how='left')

        self.data.drop(['Mask', 'Image', 'Case ID'], axis=1, inplace=True)
        self.data.dropna(inplace=True)

        # separate to input features and resulting category (last column):
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

        self.y.replace({'Alive': 1, 'Dead': 0}, inplace=True)

        scaler = MinMaxScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X))

        self.X.columns = self.X.columns