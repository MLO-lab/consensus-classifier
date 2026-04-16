import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import ray
except ImportError:
    ray = None

logger = logging.getLogger(__name__)


def ray_progress_bar(tasks, desc="Training models", verbose=True):
    """
    Show tqdm progress bar while waiting for Ray tasks to finish.
    Returns results from ray.get() in the correct completed order.
    """
    remaining = list(tasks)
    done_results = []
    pbar = tqdm(total=len(tasks), desc=desc, disable=not verbose)

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        result = ray.get(done[0])
        done_results.append(result)
        pbar.update(1)

    pbar.close()
    return done_results


def compute_required_features_per_model(
    total_features: int, n_models: int, target_coverage: float = 0.95
) -> int:
    """
    Compute the number of features that must be sampled per model to achieve
    a desired expected feature coverage across an ensemble.

    Parameters
    ----------
    total_features : int
        Total number of available features (D).
    n_models : int
        Number of ensemble models (M).
    target_coverage : float
        Desired coverage level (default: 0.95).

    Returns
    -------
    int
        Minimum required features per model (rounded up).
    """
    if not (0 < target_coverage < 1):
        raise ValueError("target_coverage must be between 0 and 1.")

    D = float(total_features)
    M = float(n_models)

    k = D * (1 - (1 - target_coverage) ** (1.0 / M))
    return int(np.ceil(k))


def intersect_features(current_features, new_features):
    return [col for col in new_features if col in current_features]

class ConsensusClassifier:
    def __init__(
        self,
        n_models=100,
        n_samples_per_model=None,
        n_features_per_model="auto",
        scaler=None,  # None | "standard" | "robust"
        keep_samples=None,
        keep_features=None,
        verbose=True,
        xgb_params_override=None,
        use_ray=False,
    ):
        if n_models < 1:
            raise ValueError(f"n_models must be at least 1, got {n_models}.")
        self.n_models = n_models
        self.n_samples_per_model = n_samples_per_model
        self.n_features_per_model = n_features_per_model
        self.scaler = scaler
        self.keep_samples = keep_samples
        self.keep_features = keep_features

        self.verbose = verbose
        self.log = logger.info if self.verbose else lambda *a, **k: None

        self.log("========== Initializing Consensus Classifier ==========")
        self.log(
            "Finished! Call the `add_train_data` method to load the training data."
        )

        self.xgb_params_override = xgb_params_override
        if self.xgb_params_override is None:
            self.xgb_params_override = {}
        self.use_ray = use_ray
        if self.use_ray and ray is None:
            logger.warning(
                "Ray not available, disabling `use_ray` in ConsensusClassifier."
            )
            self.use_ray = False
        self._reset()

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.le = None
        self.le_mapping = None
        self.n_classes = None
        self.binary = None

        self.X_test = None
        self._X_test_cache = {}
        
        self.scaler_list = []
        self.sample_subset_list = []
        self.feature_subset_list = []
        self.classifier_list = []
        
        self.probs_list = []
        self._probs_df = None
        self._pred_series = None
        self._vote_series = None
        self._all_votes_df = None
        
    def _make_scaler(self):
        if self.scaler is None:
            return None
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler type: {self.scaler}")
    
    @staticmethod
    def _to_float32(X):
        # Only convert numeric columns; leave index/columns untouched
        return X.astype(np.float32, copy=False)

    @property
    def xgb_config(self):
        if self.binary:
            scale_pos_weight = float(np.sum(self.y_train == 0)) / max(
                float(np.sum(self.y_train == 1)), 1.0
            )
            xgb_config = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "scale_pos_weight": scale_pos_weight,
            }
        else:
            xgb_config = {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": self.n_classes,
            }

        defaults = {
            "eta": 0.1,
            "gamma": 1.0,
            # l1 regularization term on weights
            "alpha": 1.0,
            # l2 regularization term on weights
            "lambda": 5.0,
            "max_depth": 4,
            "subsample": 0.8,
            "n_estimators": 200,
            "tree_method": "hist",
        }

        xgb_config.update(defaults)
        if self.xgb_params_override:
            xgb_config.update(self.xgb_params_override)

        return xgb_config

    @property
    def probs(self):
        """Return ensemble-averaged probability estimates as a DataFrame."""
        if self._probs_df is None:
            raise RuntimeError(
                "predict_proba() must be called before accessing `.probs`."
            )
        return self._probs_df

    @property
    def predictions(self):
        """Return ensemble class predictions as a Series of labels."""
        if self._pred_series is None:
            raise RuntimeError(
                "predict() must be called before accessing `.predictions`."
            )
        return self._pred_series

    @property
    def votes(self):
        """Return ensemble vote counts for the winning label per sample."""
        if self._vote_series is None:
            raise RuntimeError("predict() must be called before accessing `.votes`.")
        return self._vote_series

    @property
    def all_votes(self):
        """
        Return per-class vote counts for each sample.
        Columns are class labels.
        """
        if self._all_votes_df is None:
            raise RuntimeError(
                "predict() must be called before accessing `.all_votes`."
            )
        return self._all_votes_df

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
        if n_common < n_train_features:
            self.log(
                f"Found {n_common} out of {n_train_features} features ({percentage:.2f}%) in the test data."
            )
        if n_common == 0:
            raise ValueError(
                "No common features found between training and test data. "
                "Check that column names match."
            )
        self.log("Subsetting features of the test data...")
        X_train = X_train.loc[:, common_features].copy()
        X_test = X_test.loc[:, common_features].copy()

        if self.keep_features is not None:
            self.keep_features = [
                kf for kf in self.keep_features if kf in common_features
            ]

        if self.n_features_per_model == "auto":

            self.n_features_per_model = compute_required_features_per_model(
                total_features=X_train.shape[1],
                n_models=self.n_models,
                target_coverage=0.95,
            )

        if (
            self.n_features_per_model is None
            or self.n_features_per_model <= 0
            or self.n_features_per_model > X_train.shape[1]
        ):
            self.n_features_per_model = X_train.shape[1]

        if 0 < self.n_features_per_model <= 1.0:
            self.n_features_per_model = int(
                self.n_features_per_model * X_train.shape[1]
            )

        self.log(f"Each model will be trained on {self.n_features_per_model} features.")

        return X_train, X_test

    def add_train_data(self, X, y, y_multi=None):
        self.log("Checking training data...")
        self.X_train = self._check_data(X)
        self.y_train = y.loc[self.X_train.index].copy()
        self.y_multi = y_multi.loc[self.X_train.index].copy() if y_multi is not None else None
        self.le = LabelEncoder()
        self.le.fit(self.y_train)
        self.y_train = pd.Series(
            self.le.transform(self.y_train), index=self.y_train.index
        )
        self.n_classes = len(self.le.classes_)
        self.binary = self.n_classes == 2
        self.le_mapping = dict(zip(range(len(self.le.classes_)), self.le.classes_))

        if (
            self.n_samples_per_model is None
            or self.n_samples_per_model <= 0
            or self.n_samples_per_model > self.X_train.shape[0]
        ):
            self.n_samples_per_model = self.X_train.shape[0]

        if 0 < self.n_samples_per_model < 1.0:
            self.n_samples_per_model = int(
                self.n_samples_per_model * self.X_train.shape[0]
            )

        self.log(
            f"Loaded training data with {self.X_train.shape[0]} samples and {self.X_train.shape[1]} features."
        )
        self.log(f"Found {self.n_classes} classes in the training data.")
        self.log(f"Each model will be trained on {self.n_samples_per_model} samples.")
        self.log("Finished! Call the `add_test_data` method to load the test data.")

    def add_test_data(self, X):
        if self.X_train is None:
            raise RuntimeError("add_train_data() must be called before add_test_data().")
        self.log("Checking test data...")
        self.X_test = self._check_data(X)
        self.log(
            f"Loaded test data with {self.X_test.shape[0]} samples and {self.X_test.shape[1]} features."
        )

        self.X_train, self.X_test = self._intersect_features(self.X_train, self.X_test)

        self.X_train = self._to_float32(self.X_train)
        self.X_test = self._to_float32(self.X_test)

        self.classifier_list = []
        self.feature_subset_list = []
        self.scaler_list = []
        self._X_test_cache.clear()
        self.probs_list = []
        self._probs_df = None
        self._pred_series = None
        self._vote_series = None
        self._all_votes_df = None

        self.log("Finished! Call the `fit` method to train the classifiers.")

    if ray is not None:

        @staticmethod
        @ray.remote
        def _fit_classifier_remote(xgb_config, X, y, model_idx, use_sample_weight):
            clf = xgb.XGBClassifier(**xgb_config)
            if use_sample_weight:
                sample_weight = compute_sample_weight(class_weight="balanced", y=y)
                return clf.fit(X, y, sample_weight=sample_weight), model_idx
            return clf.fit(X, y), model_idx

    @staticmethod
    def _fit_classifier(xgb_config, X, y, use_sample_weight):
        clf = xgb.XGBClassifier(**xgb_config)
        if use_sample_weight:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
            return clf.fit(X, y, sample_weight=sample_weight)
        return clf.fit(X, y)

    def _sample_samples_per_model(self, rng):
        """
        Stratified sampling of samples for one model.
        Uses y_multi when provided, otherwise y_train.
        Ensures at least one sample per class.
        """
        y_eff = self.y_multi if self.y_multi is not None else self.y_train
        percentage = self.n_samples_per_model / len(self.y_train)

        sample_subset = []
        for label in y_eff.unique():
            idx = y_eff.index[y_eff == label].to_numpy()
            n_samples = max(1, int(len(idx) * percentage))
            n_samples = min(n_samples, len(idx))
            chosen = rng.choice(idx, size=n_samples, replace=False)
            sample_subset.extend(chosen.tolist())

        if self.keep_samples is not None:
            sample_subset = list(set(sample_subset) | set(self.keep_samples))

        return sample_subset

    def _sample_features_per_model(self, rng):
        """
        Randomly sample features for one model, optionally union with keep_features.
        """
        cols = self.X_train.columns.to_numpy()
        features = rng.choice(
            cols, size=self.n_features_per_model, replace=False
        ).tolist()

        if self.keep_features is not None:
            features = list(set(features) | set(self.keep_features))

        return features

    def fit(self, seed=None):
        """
        Train the ensemble of XGBoost classifiers by repeatedly sampling
        stratified sample subsets and feature subsets.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("add_train_data() must be called before fit().")
        if self.X_test is None:
            raise RuntimeError("add_test_data() must be called before fit().")
        rng = np.random.default_rng(seed)
        self.log("Training the classifiers...")
        self._X_test_cache.clear()

        self.sample_subset_list = []
        self.feature_subset_list = []
        self.classifier_list = []
        self.probs_list = []
        self._probs_df = None
        self._pred_series = None
        self._vote_series = None
        self._all_votes_df = None
        self.scaler_list = [None] * self.n_models

        ray_tasks = []

        iterator = range(self.n_models)
        if not self.use_ray:
            iterator = tqdm(
                iterator, desc="Training ensemble", disable=not self.verbose
            )

        for model_idx in iterator:
            # --- sample data ---
            sample_subset = self._sample_samples_per_model(rng)
            feature_subset = self._sample_features_per_model(rng)

            self.sample_subset_list.append(sample_subset)
            self.feature_subset_list.append(feature_subset)

            X_sub = self.X_train.loc[sample_subset, feature_subset]
            y_sub = self.y_train.loc[sample_subset]
            
            scaler = self._make_scaler()
            if scaler is not None:
                X_sub = pd.DataFrame(
                    scaler.fit_transform(X_sub),
                    index=X_sub.index,
                    columns=X_sub.columns,
                )

            self.scaler_list[model_idx] = scaler

            cfg = self.xgb_config.copy()
            cfg["random_state"] = (seed or 0) + model_idx

            if self.use_ray and ray is not None:
                ray_tasks.append(
                    self._fit_classifier_remote.remote(
                        xgb_config=cfg,
                        X=X_sub,
                        y=y_sub,
                        model_idx=model_idx,
                        use_sample_weight=not self.binary,
                    )
                )
            else:
                self.classifier_list.append(
                    self._fit_classifier(
                        xgb_config=cfg,
                        X=X_sub,
                        y=y_sub,
                        use_sample_weight=not self.binary,
                    )
                )

        if self.use_ray and ray is not None:
            results = [None] * self.n_models

            for clf, idx in ray_progress_bar(
                ray_tasks, desc="Training ensemble", verbose=self.verbose
            ):
                results[idx] = clf

            self.classifier_list = results
            assert all(clf is not None for clf in self.classifier_list)
        self.log("Finished! Call the `predict` method to classify the test data.")
        return self.classifier_list
    
    def _predict_proba(self):
        """
        Generate per-model probability predictions for the test set.
        Results are stored in `self.probs_list` as numpy arrays.
        """
        X_test_full = self.X_test  # alias

        self.probs_list = []
        append = self.probs_list.append  # local binding is faster

        for features, clf, scaler in zip(
            self.feature_subset_list, self.classifier_list, self.scaler_list
        ):
            key = tuple(features)
            if key not in self._X_test_cache:
                # IMPORTANT: preserve original feature order for scaler
                X_sub = X_test_full.loc[:, features]

                if scaler is not None:
                    X_sub = pd.DataFrame(
                        scaler.transform(X_sub),
                        index=X_sub.index,
                        columns=X_sub.columns,
                    )
                    assert list(X_sub.columns) == list(scaler.feature_names_in_)

                self._X_test_cache[key] = X_sub
            else:
                X_sub = self._X_test_cache[key]

            append(clf.predict_proba(X_sub))

        return self.probs_list

    @staticmethod
    def _safe_logit(p):
        """Convert probability p to logit with clipping for stability."""
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return np.log(p / (1 - p))

    @staticmethod
    def _safe_sigmoid(x):
        """Stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-x))

    def _aggregate_binary_logits(self):
        """
        Aggregate binary ensemble predictions in logit space.
        probs_list: list of (n_samples, 2) arrays.
        Returns: (n_samples, 2) array.
        """
        # extract P(positive) for each model → shape (n_models, n_samples)
        p1 = np.stack([p[:, 1] for p in self.probs_list], axis=0)

        # convert each to logits
        logits = self._safe_logit(p1)

        # mean logit across models
        mean_logits = logits.mean(axis=0)

        # convert back to probability
        p1_final = self._safe_sigmoid(mean_logits)
        p0_final = 1.0 - p1_final

        return np.stack([p0_final, p1_final], axis=1)

    def _aggregate_multiclass_logits(self):
        """
        Aggregate multiclass probabilities in logit space.
        probs_list: list of (n_samples, n_classes)
        Returns: (n_samples, n_classes)
        """
        probs = np.stack(self.probs_list, axis=0)  # shape (M, N, C)

        # clip
        probs = np.clip(probs, 1e-8, 1 - 1e-8)

        # compute logits per class
        logits = np.log(probs)

        # average logits across models
        mean_logits = logits.mean(axis=0)  # (N, C)

        # normalize back to probabilities
        exp_logits = np.exp(mean_logits)
        final = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return final

    def predict_proba(self):
        if not self.classifier_list:
            raise ValueError("No classifiers trained yet. Call fit() first.")

        if len(self.probs_list) == 0:
            self._predict_proba()

        if self.binary:
            agg_probs = self._aggregate_binary_logits()
        else:
            agg_probs = self._aggregate_multiclass_logits()

        df = pd.DataFrame(
            agg_probs,
            index=self.X_test.index,
            columns=[self.le_mapping[i] for i in range(agg_probs.shape[1])],
        )

        self._probs_df = df
        return df

    def predict(self):
        """
        Compute ensemble hard predictions and vote counts.
        Populates:
        - _all_votes_df : per-class vote counts
        - _vote_series  : votes for winning class
        - _pred_series  : predicted label
        """
        if not self.classifier_list:
            raise RuntimeError("fit() must be called before predict().")

        _ = self.predict_proba()  # ensure probs_list exists

        # (n_models, n_samples)
        hard_preds = np.stack([np.argmax(p, axis=1) for p in self.probs_list], axis=0)

        # --- per-class vote counts ---
        votes = {}
        for class_idx, class_label in self.le_mapping.items():
            votes[class_label] = (hard_preds == class_idx).sum(axis=0)

        all_votes_df = pd.DataFrame(votes, index=self.X_test.index)

        # --- winner: break vote ties via aggregated probabilities ---
        max_votes = all_votes_df.max(axis=1)
        is_tied = (all_votes_df == max_votes.values[:, None]).sum(axis=1) > 1
        pred_labels = all_votes_df.idxmax(axis=1)
        if is_tied.any():
            prob_winner = self._probs_df.loc[is_tied].idxmax(axis=1)
            pred_labels.loc[is_tied] = prob_winner
        winner_votes = all_votes_df.max(axis=1)

        self._all_votes_df = all_votes_df
        self._pred_series = pred_labels
        self._vote_series = winner_votes

        return self._pred_series

    def export_results(self, path=None):
        preds = self.predict()
        probs = self.probs
        votes = self.all_votes

        results = pd.concat(
            [preds.rename("Prediction"), votes.add_prefix("votes_"), probs],
            axis=1,
        )

        if path is not None:
            results.to_csv(path)

        return results
