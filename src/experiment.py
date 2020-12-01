from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb


from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd

from sklearn.decomposition import PCA


def gbr_model(**kwargs):  
    return GradientBoostingRegressor(**kwargs)

def ridge_model(**kwargs):
    return Pipeline([['scaler', StandardScaler()], ['ridgecv', RidgeCV(**kwargs)]])


def get_boston_dataset():
    # get dataset
    desc = datasets.load_boston()
    X, y = desc['data'], desc['target']
    return X, y, desc


def single_model_experiment(X, y, model, model_name="model", seed=1, train_size=0.9):
    np.random.seed(seed)

    X_train, X_test, \
    y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size)

    # Fit regression model
    gbr = model
    gbr.fit(X_train, np.log(y_train))
    gbr_test = np.exp(gbr.predict(X_test))
    gbr_train = np.exp(gbr.predict(X_train))

    print('train: ', np.float16(mean_squared_error(y_train, gbr_train)), \
          np.float16(r2_score(y_train, gbr_train)))
    print('test:  ', np.float16(mean_squared_error(y_test, gbr_test)), \
          np.float16(r2_score(y_test, gbr_test)))

    sb.regplot(y_train, gbr_train).get_figure().savefig(f"{model_name}-train.png")
    sb.regplot(y_test, gbr_test).get_figure().savefig(f"{model_name}-test.png")
    
    
    
def feature_importances(clf, data, path, model_name="model"):
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, data['feature_names'][sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.savefig(f"{path}/{model_name}-feature-importance.png")


def plot_gbr_deviance(gbr, path, model_name="model"):
    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((gbr.n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(gbr.staged_predict(X_test)):
        test_score[i] = gbr.loss_(y_test, np.exp(y_pred))

    plt.figure(figsize=(12, 6))
    plt.title('Deviance')
    plt.plot(np.arange(gbr.n_estimators) + 1, gbr.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(gbr.n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.ylim(-0.1, 1)
    plt.savefig(f"{path}/{model_name}-deviance.png")
    
      
        
class HiddenLoopExperiment:
    
    #params = {'loss':'huber', 'n_estimators': 1000, 'max_depth': 5, 'max_features': 1.0,
    #          'learning_rate': 0.01, 'random_state':0, 'subsample':0.5}
    default_params = {'loss':'ls', 'n_estimators': 50, 
              'max_depth': 3, 'max_features': 1.0, 
              'learning_rate': 0.5, 'random_state':0, 
              'subsample':0.75}    
    
    def __init__(self, X, y, model, model_name="model"):
        self.X = X
        self.y = y
        self.gbr_tst = []
        self.gbr_base = None
        self.model = model
        self.model_name = model_name

    
    def prepare_data(self, train_size=0.3):

        self.X_orig, self.X_new, self.y_orig, self.y_new = \
            model_selection.train_test_split(self.X, np.log(self.y), train_size=train_size)
        
        self.X_curr = self.X_orig
        self.y_curr = self.y_orig
        self.mae, self.r2 = [], []
        self.mae_orig, self.r2_orig = [], []
        self.mae_new, self.r2_new = [], []
        
        
    def add_instances(self, X, y, model, usage=0.9, adherence=0.9):
        assert callable(model)

        for sample in np.random.permutation(len(X)):
            if np.random.random() <= usage:
                m, scale = model()
                pred = m.predict([X[sample]])
                new_price = np.random.normal(pred, scale*adherence)[0]
            else:
                new_price = y[sample]

            yield sample, new_price
          
    
    def eval_m(self, gbr, X, y, mae=None, r2=None):
        gbr_pred = gbr.predict(X)
        
        mae_v = mean_absolute_error(y, gbr_pred)
        r2_v = r2_score(y, gbr_pred)
        
        if mae is not None:
            mae.append(mae_v)
        if r2 is not None:
            r2.append(r2_v)
        
        return mae_v, r2_v

    def hidden_loop_experiment(self, seed=42, adherence=0.2, usage=0.1, step=10):

        np.random.seed(seed)
        
        self.X_tr, self.X_tst, self.y_tr, self.y_tst = model_selection.train_test_split(self.X_curr, self.y_curr)

        self.gbr_base = self.model()
        self.gbr_base.fit(self.X_tr, self.y_tr)
        
        self.gbr = self.model()
        self.gbr.fit(self.X_tr, self.y_tr)
       
        self.m, self.r = self.eval_m(self.gbr, self.X_tst, self.y_tst, self.mae, self.r2)
        m_b, r_b = self.eval_m(self.gbr_base, self.X_tst, self.y_tst)

        print(self.m, self.r, m_b, r_b)
        
        self.eval_m(self.gbr, self.X_orig, self.y_orig, self.mae_orig, self.r2_orig)
        self.eval_m(self.gbr, self.X_new, self.y_new, self.mae_new, self.r2_new)


        i = 0

        for idx, pred in self.add_instances(self.X_new, self.y_new, lambda : (self.gbr, self.m), 
                                       adherence=adherence, usage=usage):
            self.X_curr = np.concatenate((self.X_curr[1:], [self.X_new[idx]]))
            self.y_curr = np.concatenate((self.y_curr[1:], [pred]))

            i = i + 1
            if i % step == 0:
                
                self.X_tr, self.X_tst, \
                self.y_tr, self.y_tst = model_selection.train_test_split(self.X_curr, self.y_curr)

                self.gbr = self.model()
                self.gbr.fit(self.X_tr, self.y_tr)

                self.m, self.r = self.eval_m(self.gbr, self.X_tst, self.y_tst, self.mae, self.r2)
                m_b, r_b = self.eval_m(self.gbr_base, self.X_tst, self.y_tst)

                print(self.m, self.r, m_b, r_b)

                self.eval_m(self.gbr, self.X_orig, self.y_orig, self.mae_orig, self.r2_orig)
                self.eval_m(self.gbr, self.X_new, self.y_new, self.mae_new, self.r2_new)

                
    def plot_results(self, path):
        plt.figure()
        sb.lineplot(range(len(self.r2)), self.r2, label="R2, dynamic data")
        plt.savefig(f"{path}/{self.model_name}-r2-dynamic.png")
        
        plt.figure()
        sb.lineplot(range(len(self.mae)), self.mae, label="MAE, dynamic data")
        plt.savefig(f"{path}/{self.model_name}-mae-dynamic.png")
        
    def save_results(self, path):
        ...