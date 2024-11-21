import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import xgboost as xgb

from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


logger = logging.getLogger(__name__)


def to_ranks(_X, method="first"):
    return pd.DataFrame(_X).rank(axis=1, method=method)


def to_z_scores(_X):
    return (_X - _X.mean(axis=0)) / _X.std(axis=0)


class ConsensusClassifier:
    def __init__(
        self,
        n_models=100,
        n_features_per_model=200,
        use_ranks=True,
        use_z_scores=False,
        verbose=True,
    ):
        self.n_models = n_models
        self.n_features_per_model = n_features_per_model
        if not use_ranks and not use_z_scores:
            logger.warning("No processing of the data selected, using raw values.")
        if use_ranks and use_z_scores:
            raise ValueError("Only one of `use_ranks` and `use_z_scores` can be True.")
        self.use_ranks = use_ranks
        self.use_z_scores = use_z_scores

        self.transform = None
        if self.use_ranks:
            self.transform = to_ranks
        if self.use_z_scores:
            self.transform = to_z_scores

        self.verbose = verbose

        if self.verbose:
            logger.info("========== Initializing Consensus Classifier ==========")
            logger.info(
                "Finished! Call the `add_train_data` method to load the training data."
            )

        self._reset()

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.le = None
        self.le_mapping = None
        self.n_classes = None
        self.binary = None

        self.X_test = None

        self.feature_subset_list = []
        self.classifier_list = []

        self.probs_list = []
        self.raw_probs = []

        self.raw_predictions = []

    @property
    def xgb_config(self):
        if self.binary:
            # scale_pos_weight = float(np.sum(self.y == 0)) / np.sum(self.y == 1)
            # scale_pos_weight *= 2
            xgb_config = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                # "scale_pos_weight": scale_pos_weight,
            }
        else:
            xgb_config = {
                "objective": "multi:softprob",
                "eval_metric": "auc",
                "num_class": 7,
            }

        xgb_config.update(
            {
                "eta": 0.05,
                "gamma": 0.0,
                # l1 regularization term on weights
                "alpha": 1.0,
                # l2 regularization term on weights
                "lambda": 5.0,
                "max_depth": 3,
                # "max_delta_step": 5.0,
                "subsample": 0.9,
            }
        )

        return xgb_config

    @property
    def probs(self):
        if self.raw_probs is None:
            raise ValueError(
                "No predictions available. Call the `predict_proba` method first."
            )
        probs = pd.DataFrame(self.raw_probs, index=self.X_test.index)
        probs.columns = probs.columns.map(self.le_mapping)
        return probs

    @property
    def predictions(self):
        if self.raw_predictions is None:
            raise ValueError(
                "No predictions available. Call the `predict` method first."
            )

        predictions = pd.Series(self.raw_predictions.mode, index=self.X_test.index)
        return predictions.map(self.le_mapping)

    @property
    def votes(self):
        if self.raw_predictions is None:
            raise ValueError(
                "No predictions available. Call the `predict` method first."
            )

        return pd.Series(self.raw_predictions.count, index=self.X_test.index)

    def _check_data(self, X):
        invalid_rows = X.index.isna()
        if any(invalid_rows):
            logger.warning(f"Found {invalid_rows.sum()} invalid rows, removing...")
        invalid_columns = X.columns.isna()
        if any(invalid_columns):
            logger.warning(
                f"Found {invalid_columns.sum()} invalid columns, removing..."
            )
        return X.loc[~invalid_rows, ~invalid_columns].copy()

    def _intersect_features(self, X_train, X_test):
        common_features = [col for col in X_test.columns if col in X_train.columns]
        n_common = len(common_features)
        n_train_features = len(X_train.columns)
        percentage = 100 * (n_common / n_train_features)
        if (n_common < n_train_features) and self.verbose:
            logger.info(
                f"Found {n_common} out of {n_train_features} features ({percentage:.2f}%) in the test data."
            )

        if self.verbose:
            logger.info("Subsetting features of the test data...")
        X_train = X_train.loc[:, common_features].copy()
        X_test = X_test.loc[:, common_features].copy()

        if (
            self.n_features_per_model is None
            or self.n_features_per_model > self.X_train.shape[1]
        ):
            self.n_features_per_model = self.X_train.shape[1]

        return X_train, X_test

    def add_train_data(self, X, y):
        # self._reset()
        if self.verbose:
            logger.info("Checking training data...")
        self.X_train = self._check_data(X)
        self.y_train = y.copy()
        self.le = LabelEncoder()
        self.le.fit(self.y_train)
        self.y_train = self.le.transform(self.y_train)
        self.n_classes = len(self.le.classes_)
        self.binary = self.n_classes == 2
        self.le_mapping = dict(zip(range(len(self.le.classes_)), self.le.classes_))
        if self.verbose:
            logger.info(
                f"Loaded training data with {self.X_train.shape[0]} samples and {self.X_train.shape[1]} features."
            )
            logger.info(f"Found {self.n_classes} classes in the training data.")
            logger.info(
                "Finished! Call the `add_test_data` method to load the test data."
            )

    def add_test_data(self, X):
        # self._reset()
        if self.verbose:
            logger.info("Checking training data...")
        self.X_test = self._check_data(X)
        if self.verbose:
            logger.info(
                f"Loaded test data with {self.X_test.shape[0]} samples and {self.X_test.shape[1]} features."
            )

        self.X_train, self.X_test = self._intersect_features(self.X_train, self.X_test)

        if self.verbose:
            logger.info("Finished! Call the `fit` method to train the classifiers.")

    def _fit_classifier(self, X, y, seed, model_idx):
        seed += model_idx
        xgb_config = self.xgb_config
        xgb_config["seed"] = seed
        xgb_clf = xgb.XGBClassifier(**xgb_config)
        return xgb_clf.fit(
            X, y, sample_weight=compute_sample_weight(class_weight="balanced", y=y)
        )

    def fit(self, seed=None):
        rng = np.random.default_rng(seed)
        if self.verbose:
            logger.info("Training the classifiers...")
        self.feature_subset_list = []
        self.classifier_list = []
        for model_idx in tqdm(range(self.n_models)):
            feature_subset = rng.choice(
                self.X_train.columns, size=self.n_features_per_model, replace=False
            )
            self.feature_subset_list.append(feature_subset)
            X_train_sub = self.X_train.loc[:, feature_subset]
            if self.transform is not None:
                X_train_sub = self.transform(X_train_sub)
            classifier = self._fit_classifier(
                X_train_sub, self.y_train, seed, model_idx
            )
            self.classifier_list.append(classifier)
        if self.verbose:
            logger.info(
                "Finished! Call the `predict` method to classify the test data."
            )
        return self.classifier_list

    def _predict_proba(self):
        self.probs_list = []
        for feature_subset, classifier in zip(
            self.feature_subset_list, self.classifier_list
        ):
            X_test_sub = self.X_test.loc[:, feature_subset]
            if self.transform is not None:
                X_test_sub = self.transform(X_test_sub)
            probs = classifier.predict_proba(X_test_sub)
            self.probs_list.append(probs)

        return self.probs_list

    def predict_proba(self):
        if not self.classifier_list:
            raise ValueError("No classifiers trained yet. Call the `fit` method first.")

        if len(self.probs_list) == 0:
            self._predict_proba()
        agg_probs = np.mean(np.array(self.probs_list), axis=0)
        agg_probs = np.exp(agg_probs) / np.exp(agg_probs).sum(axis=1)[:, None]
        self.raw_probs = agg_probs
        return self.probs

    def predict(self):
        if not self.classifier_list:
            raise ValueError("No classifiers trained yet. Call the `fit` method first.")

        self.raw_predictions = []
        if len(self.probs_list) == 0:
            self._predict_proba()

        predictions = np.array([np.argmax(probs, axis=1) for probs in self.probs_list])
        self.raw_predictions = stats.mode(predictions, axis=0)
        return self.predictions

    def _label_frequency(self, test=False):
        y = self.y_train
        if test:
            y = self.raw_predictions
        return np.unique(y, return_counts=True)

    def _frequency_to_df(self, label, frequency):
        df = pd.DataFrame({"Cluster": label, "Count": frequency})
        df["Cluster"] = df["Cluster"].map(self.le_mapping)
        df["Fraction"] = df["Count"] / df["Count"].sum()
        return df

    def get_label_frequency(self):
        train_freq_df = self._frequency_to_df(*self._label_frequency(test=False))
        train_freq_df["Type"] = "Train"
        test_freq_df = self._frequency_to_df(*self._label_frequency(test=True))
        test_freq_df["Type"] = "Test"
        return pd.concat([train_freq_df, test_freq_df])

    def export_results(self, path=None):
        self.predict_proba()
        self.predict()
        results = pd.concat([self.probs, self.predictions, self.votes], axis=1)
        results = results.rename(columns={0: "Cluster", 1: "Votes"})
        if path is not None:
            results.to_csv(path)
        return results
