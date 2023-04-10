import os
import time
import pickle as pkl
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from feature_importance import dec_tree_feat_importance, log_reg_feat_importance, naive_bayes_feat_importance, cat_boost_feat_importance
import numpy as np
import pandas as pd

class Model:
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        self.models_path = models_path
        self.force_recompute = force_recompute
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_cols = feature_cols
        self.get_feat_importance = get_feat_importance
        self.trained_model = None

    def set_trained_model(self):
        if os.path.exists(os.path.join(self.models_path, self.model_file)):
            with open(os.path.join(self.models_path, self.model_file), 'rb') as f:
                self.trained_model = pkl.load(f)

    def train(self, model_type, cat_features=None):
        if self.trained_model is None or self.force_recompute:
            start = time.time()
            if cat_features is not None:
                model_type.fit(self.X_train[self.feature_cols], self.y_train, cat_features)
            else:
                model_type.fit(self.X_train[self.feature_cols], self.y_train)
            end = time.time()
            print("Model took {} seconds to train!".format(end - start))
            with open(os.path.join(self.models_path, self.model_file), 'wb') as f:
                pkl.dump(model_type, f)
            self.trained_model = model_type
    
    def test(self, catboost=False):
        if self.trained_model is not None:
            if catboost:
                y_pred = self.trained_model.predict(self.X_test[self.feature_cols]) == 'True'
            else:
                y_pred = self.trained_model.predict(self.X_test[self.feature_cols])
            print("Confusion matrix is: {}".format(confusion_matrix(self.y_test, y_pred)))
            print("F1 score is {}".format(f1_score(self.y_test, y_pred)))
            print("Accuracy is {}".format(accuracy_score(self.y_test, y_pred)))
        else:
            raise ValueError("There is no trained model for this type of classifier. Try to set force_recompute to True!")


class MLPClassifierModel(Model):
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_file = 'mlp.pkl'
        self.set_trained_model()

    def train(self):
        model_mlp = MLPClassifier(solver='lbfgs',
                                    hidden_layer_sizes= [10],
                                    random_state=42,
                                    learning_rate='adaptive')
        super().train(model_mlp)


class DecisionTreeModel(Model):
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_file = 'dectree.pkl'
        self.set_trained_model()

    def train(self):
        model_dt = DecisionTreeClassifier(class_weight=None, criterion='gini',
                                    min_impurity_decrease=0.0,
                                    min_samples_leaf=100,
                                    min_samples_split=2,
                                    min_weight_fraction_leaf=0.0,
                                    random_state=42,
                                    splitter='best')
        super().train(model_dt)
        if self.get_feat_importance:
                dec_tree_feat_importance(self.trained_model)


class LogisticRegressionModel(Model):
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_file = 'logreg.pkl'
        self.set_trained_model()

    def train(self):
        model_lr = LogisticRegression(max_iter=1000, random_state=42)
        super().train(model_lr)
        if self.get_feat_importance:
                log_reg_feat_importance(self.trained_model)

        
class NaiveBayesModel(Model):
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_file = 'bayes.pkl'
        self.set_trained_model()
    
    def train(self):
        model_nb = GaussianNB(priors=None, var_smoothing=1e-09)
        super().train(model_nb)
        if self.get_feat_importance:
            naive_bayes_feat_importance(self.trained_model, self.X_train[self.feature_cols], self.y_train)


class CatBoostModel(Model):
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_file = 'catboost.pkl'
        self.set_trained_model()

    def train(self):
        model_catb = CatBoostClassifier(iterations=1000,
                        learning_rate=0.1,
                        od_type = "Iter",
                        od_wait = 100,
                        random_state=42,
                        depth=8)
        super().train(model_catb)
        if self.get_feat_importance:
            cat_boost_feat_importance(self.trained_model)

class CatBoostClusterModel(Model):
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_file = 'catboost_cluster.pkl'
        self.set_trained_model()
    
    def train(self):
        model_catb_cluster = CatBoostClassifier(iterations=1000,
                            learning_rate=0.1,
                            od_type = "Iter",
                            od_wait = 100,
                            random_state=42,
                            depth=8)
        super().train(model_catb_cluster, cat_features=['Cluster'])
        if self.get_feat_importance:
            cat_boost_feat_importance(self.trained_model)


class SpecializedCatBoostModel(Model):
    """
    This class is used to create a custom trading strategy based on multiple
    specialized models per cluster. There are as many models as clusters, and
    each model is trained/predicts only on data from a specific cluster.
    """
    def __init__(self, models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False):
        super().__init__(models_path, force_recompute, X_train, X_test, y_train, y_test, feature_cols, get_feat_importance=False)
        self.model_files = {}
        for cluster in self.X_train['Cluster'].unique():
            self.model_files[cluster] = "catboost_specialized_{}.pkl".format(cluster)

            self.trained_models = {}
            if os.path.exists(os.path.join(self.models_path, self.model_files[cluster])):
                with open(os.path.join(self.models_path, self.model_files[cluster]), 'rb') as f:
                    self.trained_models[cluster] = pkl.load(f)
        
    def train(self):
        for cluster in self.X_train['Cluster'].unique():
            if self.trained_models.get(cluster, None) is None or self.force_recompute:
                specialized_model = CatBoostClassifier(iterations=1000,
                                    learning_rate=0.1,
                                    od_type = "Iter",
                                    od_wait = 100,
                                    random_state=42,
                                    depth=8)
                start = time.time()
                cluster_mask = self.X_train['Cluster'] == cluster
                specialized_model.fit(self.X_train[cluster_mask][self.feature_cols],
                                    self.y_train[cluster_mask])
                end = time.time()

                print("Model took {} seconds to train!".format(end - start))
                with open(os.path.join(self.models_path, self.model_files[cluster]), 'wb') as f:
                    pkl.dump(specialized_model, f)
        
            self.trained_models[cluster] = specialized_model

    def predict(self, X):
        predictions = pd.DataFrame(index=X.index)
        predictions['Positions'] = 0
        for cluster, model in self.trained_models.items():
            cluster_mask = X['Cluster'] == cluster
            cluster_inputs = X[cluster_mask]
            predictions['Positions'].loc[cluster_mask] = model.predict(cluster_inputs)
        predictions['Positions'] = predictions['Positions'].replace('True', True)
        predictions['Positions'] = predictions['Positions'].replace('False', False)
        return np.array(predictions['Positions'])

    def test(self, catboost=True):
            y_pred = self.predict(self.X_test[self.feature_cols + ['Cluster']])
            print("Confusion matrix is: {}".format(confusion_matrix(self.y_test, y_pred)))
            print("F1 score is {}".format(f1_score(self.y_test, y_pred)))
            print("Accuracy is {}".format(accuracy_score(self.y_test, y_pred)))
