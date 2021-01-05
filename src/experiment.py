from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def gbr_model(**kwargs):
    """
    Creates a boosted trees regressor trained with a gradient descent method.

    Uses `GradientBoostingRegressor` from `scikit-learn`

    :param kwargs: parameters for `GradientBoostingRegressor`
    :return: an instance of regressor
    """
    return GradientBoostingRegressor(**kwargs)


def ridge_model(**kwargs):
    """
    Creates a pipeline of a data scaler and a ridge regression.
    Scaler transforms data to zero mean and unit variance.
    Hyperparameters or the regression are tuned via cross-validation

    Uses `StandardScaler` and `RidgeCV` from `scikit-learn`
    :param kwargs: parameters for `RidgeCV`
    :return: an instance of `Pipeline`
    """
    return Pipeline([['scaler', StandardScaler()], ['ridgecv', RidgeCV(**kwargs)]])


def get_boston_dataset():
    """
    Return a Boston dataset from `scikit-learn`
    :return: X, y and description
    """
    # get dataset
    desc = datasets.load_boston()
    X, y = desc['data'], desc['target']
    return X, y, desc


def single_model_experiment(X, y, model, model_name="model", seed=1, train_size=0.9):
    """
    Trains a `model` on the given dataset `X` with the target `y`.

    Saves regression plot to `{model_name}-train-test.png`

    :param X: dataset to train and test on (uses split via `train_size`)
    :param y: target variable for regression
    :param model: a class/constructor of the model, should be `callable` which returns and instance of the model with a `fit` method
    :param model_name: a filename for the model, may include path
    :param seed:
    :param train_size:
    :return: None
    """
    np.random.seed(seed)

    X_train, X_test, \
    y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size)

    # Fit regression model
    gbr = model()
    gbr.fit(X_train, np.log(y_train))
    gbr_test = np.exp(gbr.predict(X_test))
    gbr_train = np.exp(gbr.predict(X_train))

    print('train: ', np.float16(mean_squared_error(y_train, gbr_train)),
          np.float16(r2_score(y_train, gbr_train)))
    print('test:  ', np.float16(mean_squared_error(y_test, gbr_test)),
          np.float16(r2_score(y_test, gbr_test)))

    plt.figure()
    plt.title = "Regression plot"
    sb.regplot(y_train, gbr_train, label="Train")
    sb.regplot(y_test, gbr_test, label="Test")
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('prediction')
    plt.savefig(f"{model_name}-train-test.png")


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


def plot_gbr_deviance(X_test, y_test, gbr, path, model_name="model"):
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
    """
    The main experiment for hidden loops paper
    See details in the paper.

    In short.

    Creates a feedback loop on a regression problem (e.g. Boston housing).
    Some of the model predictions are adhered to by users and fed back into the model as training data.
    Users add a normally distributed noise to the log of the target variable (price).
    Uses a sliding window to retrain the model on new data.

    """


    # params = {'loss':'huber', 'n_estimators': 1000, 'max_depth': 5, 'max_features': 1.0,
    #          'learning_rate': 0.01, 'random_state':0, 'subsample':0.5}
    default_params = {'loss':'ls', 'n_estimators': 50, 
                      'max_depth': 3, 'max_features': 1.0,
                      'learning_rate': 0.5, 'random_state':0,
                      'subsample':0.75}
    
    def __init__(self, X, y, model, model_name="model"):
        """
        Creates an instance of the experiment

        :param X: a dataset for regression
        :param y: target variable
        :param model: a class/constructor of the model, should be `callable` which returns and instance of the model with a `fit` method
        :param model_name: a filename to use for figures
        """
        self.X = X
        self.y = y
        self.gbr_tst = []
        self.gbr_base = None
        self.model = model
        self.model_name = model_name

    def prepare_data(self, train_size=0.3):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        self.X_orig, self.X_new, self.y_orig, self.y_new = \
            model_selection.train_test_split(self.X, np.log(self.y), train_size=train_size)
        
        self.X_curr = self.X_orig
        self.y_curr = self.y_orig
        self.mae, self.r2 = [], []
        self.mae_orig, self.r2_orig = [], []
        self.mae_new, self.r2_new = [], []

    def _add_instances(self, X, y, usage=0.9, adherence=0.9):
        """
        This is a generator function (co-routine) for the sliding window loop.
        Works as follows.

        Called once when the loop is initialized.
        Python creates a generator that returns any values provided from this method.
        The method returns the next value via `yield` and continues when `next()` is called on the generator.

        `X` and `y` are set on the first invocation.

        :param X:
        :param y:
        :param usage: how closely users adhere to predictions: `0` means exactly
        :param adherence: share of users to follow prediction
        :return: yields a new sample index from `X`, new price - from `y` or as model predicted
        """

        for sample in np.random.permutation(len(X)):
            if np.random.random() <= usage:
                pred = self.gbr.predict([X[sample]])
                new_price = np.random.normal(pred, self.m*adherence)[0]
            else:
                new_price = y[sample]

            yield sample, new_price

    def eval_m(self, model, X, y, mae=None, r2=None):
        gbr_pred = model.predict(X)
        
        mae_v = mean_absolute_error(y, gbr_pred)
        r2_v = r2_score(y, gbr_pred)
        
        if mae is not None:
            mae.append(mae_v)
        if r2 is not None:
            r2.append(r2_v)
        
        return mae_v, r2_v

    def hidden_loop_experiment(self, seed=42, adherence=0.2, usage=0.1, step=10):
        """
        Main method of the experiment

        :param seed:
        :param adherence: how closely users follow model predictions
        :param usage: how often users follow predictions
        :param step: number of steps the model is retrained
        :return: None
        """
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

        for idx, pred in self._add_instances(self.X_new, self.y_new,
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
        """
        Saves figures to the `path` specified: mean absolute error (MAE) and R2 over iterations

        :param path: base folder for plots
        :return:
        """
        plt.figure()
        sb.lineplot(range(len(self.r2)), self.r2, label="R2, dynamic data")
        plt.legend()
        plt.xlabel('rounds')
        plt.savefig(f"{path}/{self.model_name}-r2-dynamic.png")
        
        plt.figure()
        sb.lineplot(range(len(self.mae)), self.mae, label="MAE, dynamic data")
        plt.legend()
        plt.xlabel('rounds')
        plt.savefig(f"{path}/{self.model_name}-mae-dynamic.png")
        
    def save_results(self, path):
        """
        Not implemented

        :param path:
        :return:
        """
        ...
