#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import pprint
import unittest
from omnixai-community.utils.misc import set_random_seed
from omnixai-community.explainers.tabular import SensitivityAnalysisTabular
from omnixai-community.tests.explainers.tasks import TabularRegression
from omnixai-community.explanations.base import ExplanationBase


class TestSensitivity(unittest.TestCase):
    def test_1(self):
        set_random_seed()
        task = TabularRegression().train_boston_continuous()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = SensitivityAnalysisTabular(training_data=task.train_data, predict_function=predict_function)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())

        s = explanations.to_json()
        e = ExplanationBase.from_json(s)
        self.assertEqual(s, e.to_json())
        e.plotly_plot()

    def test_2(self):
        set_random_seed()
        task = TabularRegression().train_boston_continuous()
        predict_function = lambda z: task.model.predict(task.transform.transform(z))
        explainer = SensitivityAnalysisTabular(training_data=task.train_data, predict_function=predict_function)

        base_folder = os.path.dirname(os.path.abspath(__file__))
        directory = f"{base_folder}/../../datasets/tmp"
        explainer.save(directory=directory)
        explainer = SensitivityAnalysisTabular.load(directory=directory)
        explanations = explainer.explain()
        pprint.pprint(explanations.get_explanations())


if __name__ == "__main__":
    unittest.main()
