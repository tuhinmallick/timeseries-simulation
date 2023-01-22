import warnings, pathlib, os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.smoother import ConvolutionSmoother

utils_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(utils_location) not in sys.path:
    sys.path.append(os.path.realpath(utils_location))
# self-defined modules
from _misc import config_parser


# add any potential models to be recognized here (sel_estimator, est_estimator)


class PipelineModel:
    def __init__(self, hps: dict, estimator, multiout_wrapper=True, **kwargs):
        self.hps = hps
        self.estimator = estimator
        self.multiout_wrapper = multiout_wrapper
        self.kwargs = kwargs
        self.model = self.get_model(hps, estimator, multiout_wrapper, **kwargs)

    @staticmethod
    def get_model(hps: dict, estimator, multiout_wrapper=True, **kwargs):
        # reg_alpha: determine the regression at which quantile
        # In order to use hpo.py, must have hps argument
        # parse dict
        sca_hps = config_parser("sca", hps)
        sel_hps = config_parser("sel", hps)
        est_hps = config_parser("est", hps)

        # parse selector
        steps = []
        if sca_hps:
            sca_estimator = eval(sca_hps["estimator"])
            del sca_hps["estimator"]
            steps.append(("sca", sca_estimator))
        if sel_hps:
            sel_estimator = sel_hps["estimator"]["type"]
            sel_estimator = eval(f"{sel_estimator}")
            del sel_hps["estimator"]
            steps.append(("sel", RFE(sel_estimator, **sel_hps)))

        # turn on/off MultiOutputRegressor
        final_est = estimator(**est_hps)
        if multiout_wrapper:
            final_est = MultiOutputRegressor(final_est)

        steps.append(("est", final_est))
        return Pipeline(steps=steps, **kwargs)

    def fit(self, X, y, **fit_params):
        """fit the model.

        Args:
            X (iterable): Training data.
            y (iterable): Training targets.
            **fit_params (dict of string): Parameters passed to the ``fit`` method of each step, where each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.

        Returns:
            self (object): Pipeline with fitted steps.
        """
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    @staticmethod
    def get_gt_pred_data(
        y_tra, tra_pred, y_tes, tes_pred, gt, horizon, targets, out_pred=None, ci=None
    ):
        # Generate gt + prediction df file
        tra_pred = pd.DataFrame(
            tra_pred, columns=targets + "_pred", index=y_tra.index.shift(horizon)
        )
        tes_pred = pd.DataFrame(
            tes_pred, columns=targets + "_pred", index=y_tes.index.shift(horizon)
        )
        outputdf = pd.concat([gt[targets], tra_pred.append(tes_pred)], axis=1)
        if out_pred is not None:
            out_pred = pd.DataFrame(
                np.array(out_pred),
                columns=targets + "_pred",
                index=outputdf.index[-horizon:].shift(horizon),
            )
        outputdf = outputdf.append(out_pred)
        if ci is not None:
            outputdf = pd.concat([outputdf, ci], axis=1)
        return outputdf

    def get_quantile_regression_pred_interval(
        self, X_tra, y_tra, X_tes, y_tes=None, low=0.05, high=0.95
    ):

        # get hps and add relevant quantile setting
        hps = self.hps.copy()
        hps["est__objective"] = "quantile"
        hps["est__metric"] = "quantile"

        # get prediciton of different quantile models
        tra_pred_li, tes_pred_li = [], []
        for quan in [low, high]:
            # build quantile model
            hps["est__alpha"] = quan
            # quan_model = deepcopy(self.model)
            quan_model = Pipeline(steps=self.model.steps[:-1])
            tmp_tuple = PipelineModel.get_model(
                hps, self.estimator, self.multiout_wrapper, **self.kwargs
            ).steps[-1]
            # fit quantile model only on last step
            operand = X_tra
            for transformer in quan_model:
                operand = transformer.transform(operand)
            # append last step and fit it
            quan_model.steps.append(tmp_tuple)
            quan_model[-1].fit(operand, y_tra)

            # get prediction
            tra_pred = quan_model.predict(X_tra)
            tra_pred_li.append(tra_pred)
            if X_tes is not None:
                tes_pred = quan_model.predict(X_tes)
                tes_pred_li.append(tes_pred)
        # reform it and get CI
        # sort the prediction just in case. But theoretically it should not be necessary.
        tra_pred_li, tes_pred_li = np.array(tra_pred_li), np.array(tes_pred_li)
        tmp = np.concatenate(
            [np.sort(tra_pred_li, axis=0), np.sort(tes_pred_li, axis=0)], axis=1
        )
        ci_lower, ci_upper = tmp[0, :, :], tmp[-1, :, :]
        return ci_lower, ci_upper

    def get_bootstrap_pred_interval(
        self,
        X,
        y,
        n_samples: int,
        train_size: int,
        block_length: int,
        bootstrap_type: str = "cbb",
        low=0.05,
        high=0.95,
        seed: int = 42,
        model_mode="last_step_tune",
    ):
        assert X.ndim <= 2
        assert y.ndim <= 2
        assert model_mode in ["last_step_tune", "all_tune"]
        # build bootstrapper
        bts = BootstrappingWrapper(
            Smoother=ConvolutionSmoother(window_len=8, window_type="ones"),
            bootstrap_type=bootstrap_type,
            block_length=block_length,
        )
        # bootstrap data: BootstrappingWrapper only accepts 1D data
        # The goal here is to generate different dataset.
        data = np.concatenate([X, y], axis=1)
        bts_samples = []
        for i in range(data.shape[1]):
            ## reset seed in every iteration
            np.random.seed(seed)
            ## smoother doesn't accept nan values
            tmp = data[:, i]
            mask = np.isnan(tmp)
            tmp[mask] = np.random.normal(
                tmp[~mask].mean(), tmp[~mask].std(), size=mask.sum()
            )
            ## sampling
            bts_samples.append(bts.sample(tmp, n_samples=n_samples))
        bts_samples = np.stack(bts_samples, axis=-1)
        # (optional) plot for debug and insight
        # if np.isnan(data).sum() == 0:
        #     np.random.seed(datetime.datetime.now().microsecond)
        #     idx = np.random.choice(data.shape[1], 3, replace=False)
        #     smoother = ConvolutionSmoother(window_len=8, window_type='ones')
        #     smoother.smooth(data)
        #     for i in idx:
        #         plt.scatter(range(len(data)), smoother.data[...,i], c='black', s=4)
        #         plt.fill_between(range(len(data)), bts_samples[...,i].min(0), bts_samples[...,i].max(0), alpha=0.3, color='orange')
        #         plt.title(i)
        #         plt.show()

        # split the data
        ycols = 1 if y.ndim == 1 else y.shape[1]
        Xbts, ybts = bts_samples[..., :-ycols], bts_samples[..., -ycols:]
        X_tra, y_tra = Xbts[:, :train_size, :], ybts[:, :train_size, :]
        # fit model
        pred, model_li = [], []
        with warnings.catch_warnings(record=False) as cw:
            warnings.simplefilter("ignore")
            for i in tqdm(range(X_tra.shape[0])):
                if model_mode == "all_tune":
                    model = PipelineModel.get_model(
                        self.hps, self.estimator, self.multiout_wrapper, **self.kwargs
                    )
                    model.fit(X_tra[i], y_tra[i])
                elif model_mode == "last_step_tune":
                    model = Pipeline(steps=self.model.steps[:-1])
                    tmp_tuple = PipelineModel.get_model(
                        self.hps, self.estimator, self.multiout_wrapper, **self.kwargs
                    ).steps[-1]
                    # fit quantile model only on last step
                    tmp = X_tra[i]
                    for transformer in model:
                        tmp = transformer.transform(tmp)
                    # append last step and fit it
                    model.steps.append(tmp_tuple)
                    model[-1].fit(tmp, y_tra[i])
                pred.append(model.predict(X))
                model_li.append(model)

        # sort prediction and get CI
        pred = np.sort(np.array(pred), axis=0)
        idl, idm, idu = (
            int(pred.shape[0] * low),
            pred.shape[0] // 2,
            int(pred.shape[0] * high),
        )
        ci_lower, median, ci_upper = pred[idl], pred[idm], pred[idu]
        return {
            "ci_lower": ci_lower,
            "median": median,
            "ci_upper": ci_upper,
            "bootstrapped_models": model_li,
        }

    def get_feature_importance(
        self, in_feature_names, targets_names=[], normalize=False
    ):
        # input_cols should be X_tra.columns
        if "sel" in self.model.named_steps:
            support_cols = in_feature_names[self.model["sel"].get_support()]
            # Commenting out the below because it doesn't seem to be used
            # sel_fi = pd.DataFrame(self.model['sel'].estimator_.coef_.T,
            #                 index=support_cols, columns=targets_names)
        else:
            support_cols = in_feature_names
        # get feature importances
        fi = []
        for i in range(len(targets_names)):
            est = self.model["est"]
            if self.multiout_wrapper:
                est = est.estimators_[i]
            if hasattr(est, "feature_importances_"):
                fi.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                fi.append(est.coef_)
        fi = pd.DataFrame(fi, index=targets_names, columns=support_cols).T

        if normalize:
            fi /= fi.sum()
        return fi

    def binary_performances(self, y_prob, thresh=0.5, labels=["Positives", "Negatives"]):

        shape = y_prob.shape
        if len(shape) > 1:
            if shape[1] > 2:
                raise ValueError("A binary class problem is required")
            else:
                y_prob = y_prob[:, 1]

        plt.figure(figsize=[15, 4])

        # 1 -- Confusion matrix
        cm = confusion_matrix(self, (y_prob > thresh).astype(int))

        plt.subplot(131)
        ax = sns.heatmap(
            cm, annot=True, cmap="Blues", cbar=False, annot_kws={"size": 14}, fmt="g"
        )
        cmlabels = [
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives",
        ]
        for i, t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title("Confusion Matrix", size=15)
        plt.xlabel("Predicted Values", size=13)
        plt.ylabel("True Values", size=13)

        # 2 -- Distributions of Predicted Probabilities of both classes
        plt.subplot(132)
        plt.hist(
            y_prob[self == 1],
            density=True,
            bins=25,
            alpha=0.5,
            color="green",
            label=labels[0],
        )
        plt.hist(
            y_prob[self == 0],
            density=True,
            bins=25,
            alpha=0.5,
            color="red",
            label=labels[1],
        )
        plt.axvline(thresh, color="blue", linestyle="--", label="Boundary")
        plt.xlim([0, 1])
        plt.title("Distributions of Predictions", size=15)
        plt.xlabel("Positive Probability (predicted)", size=13)
        plt.ylabel("Samples (normalized scale)", size=13)
        plt.legend(loc="upper right")

        # 3 -- ROC curve with annotated decision point
        fp_rates, tp_rates, _ = roc_curve(self, y_prob)
        roc_auc = auc(fp_rates, tp_rates)
        plt.subplot(133)
        plt.plot(
            fp_rates,
            tp_rates,
            color="orange",
            lw=1,
            label="ROC curve (area = %0.3f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")
        tn, fp, fn, tp = list(cm.ravel())
        plt.plot(
            fp / (fp + tn), tp / (tp + fn), "bo", markersize=8, label="Decision Point"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", size=13)
        plt.ylabel("True Positive Rate", size=13)
        plt.title("ROC Curve", size=15)
        plt.legend(loc="lower right")
        plt.subplots_adjust(wspace=0.3)
        plt.show()

        tn, fp, fn, tp = list(cm.ravel())
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)
        results = {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": F1,
            "AUC": roc_auc,
        }

        prints = [f"{kpi}: {round(score, 3)}" for kpi, score in results.items()]
        prints = " | ".join(prints)
        print(prints)

        return results
