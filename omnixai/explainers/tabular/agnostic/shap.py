#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The SHAP explainer for tabular data.
"""
import shap
import numpy as np
from typing import Callable, List

from ..base import TabularExplainer
from ....data.tabular import Tabular
from ....explanations.tabular.feature_importance import FeatureImportance


class ShapTabular(TabularExplainer):
    """
    The SHAP explainer for tabular data.
    If using this explainer, please cite the original work: https://github.com/slundberg/shap.
    """

    explanation_type = "local"
    alias = ["shap"]

    def __init__(
            self,
            training_data: Tabular,
            predict_function: Callable,
            mode: str = "classification",
            ignored_features: List = None,
            **kwargs
    ):
        """
        :param training_data: The data used to initialize a SHAP explainer. ``training_data``
            can be the training dataset for training the machine learning model. If the training
            dataset is large, please set parameter ``nsamples``, e.g., ``nsamples = 100``.
        :param predict_function: The prediction function corresponding to the model to explain.
            When the model is for classification, the outputs of the ``predict_function``
            are the class probabilities. When the model is for regression, the outputs of
            the ``predict_function`` are the estimated values.
        :param mode: The task type, e.g., `classification` or `regression`.
        :param ignored_features: The features ignored in computing feature importance scores.
        :param kwargs: Additional parameters to initialize `shap.KernelExplainer`, e.g., ``nsamples``.
            Please refer to the doc of `shap.KernelExplainer`.
        """
        super().__init__(training_data=training_data, predict_function=predict_function, mode=mode, **kwargs)
        self.ignored_features = set(ignored_features) if ignored_features is not None else set()
        if self.target_column is not None:
            assert self.target_column not in ignored_features, \
                f"The target column {self.target_column} cannot be in the ignored feature list."

        if "nsamples" in kwargs:
            data = shap.sample(self.data, nsamples=kwargs["nsamples"])
        else:
            data = self.data
        self.explainer = shap.KernelExplainer(
            self.predict_fn, data, link="logit" if mode == "classification" else "identity", **kwargs
        )

    def explain(self, X, y=None, **kwargs) -> FeatureImportance:
        """
        Generates the feature-importance explanations for the input instances.

        :param X: A batch of input instances. When ``X`` is `pd.DataFrame`
            or `np.ndarray`, ``X`` will be converted into `Tabular` automatically.
        :param y: A batch of labels to explain. For regression, ``y`` is ignored.
            For classification, the top predicted label of each instance will be explained
            when ``y = None``.
        :param kwargs: Additional parameters for `shap.KernelExplainer.shap_values`.
        :return: The feature-importance explanations for all the input instances.
        """
        X = self._to_tabular(X).remove_target_column()
        explanations = FeatureImportance(self.mode)
        instances = self.transformer.transform(X)
        shap_values = self.explainer.shap_values(instances, **kwargs)

        if self.mode == "classification":
            if y is not None:
                if type(y) == int:
                    y = [y for _ in range(len(instances))]
                else:
                    assert len(instances) == len(y), (
                        f"Parameter ``y`` is a {type(y)}, the length of y "
                        f"should be the same as the number of instances in X."
                    )
            else:
                prediction_scores = self.predict_fn(instances)
                y = np.argmax(prediction_scores, axis=1)
        else:
            y = None

        if len(self.ignored_features) == 0:
            valid_indices = list(range(len(self.feature_columns)))
        else:
            valid_indices = [i for i, f in enumerate(self.feature_columns) if f not in self.ignored_features]

        for i, instance in enumerate(instances):
            df = X.iloc(i).to_pd()
            feature_values = [df[self.feature_columns[i]].values[0] for i in valid_indices]
            feature_names = [self.feature_columns[i] for i in valid_indices]
            if self.mode == "classification":
                label = y[i]
                importance_scores = shap_values[label][i]
                explanations.add(
                    instance=df,
                    target_label=label,
                    feature_names=feature_names,
                    feature_values=feature_values,
                    importance_scores=importance_scores[valid_indices],
                    sort=True,
                )
            else:
                explanations.add(
                    instance=df,
                    target_label=None,
                    feature_names=feature_names,
                    feature_values=feature_values,
                    importance_scores=shap_values[i][valid_indices],
                    sort=True,
                )
        return explanations
