#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The OmniXAI dashboard.
"""
import copy
import json
import os
import sys

from IPython import get_ipython
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
import warnings

import dash_bootstrap_components as dbc
import omnixai_community.visualization.callbacks.data_exp
import omnixai_community.visualization.callbacks.global_exp
import omnixai_community.visualization.callbacks.local_exp
import omnixai_community.visualization.callbacks.prediction_exp
import omnixai_community.visualization.callbacks.whatif_exp
import omnixai_community.visualization.state as board

from ..explainers.tabular import TabularExplainer
from .layout import create_banner, create_layout
from .pages.data_exp import create_data_explanation_layout
from .pages.global_exp import create_global_explanation_layout
from .pages.local_exp import create_local_explanation_layout
from .pages.prediction_exp import create_prediction_explanation_layout
from .pages.whatif_exp import create_what_if_layout

board.init()


_in_ipython = get_ipython() is not None
_in_colab = "google.colab" in sys.modules
_use_jupyter_dash = _in_ipython and _in_colab

META_TAGS = [
    {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1",
    }
]
EXTERNAL_STYLESHEETS = [dbc.themes.BOOTSTRAP]
DASHBOARD_TITLE = "OmniXAI"

if _use_jupyter_dash:
    app = JupyterDash(
        __name__,
        meta_tags=META_TAGS,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        title=DASHBOARD_TITLE,
    )
else:
    app = dash.Dash(
        __name__,
        meta_tags=META_TAGS,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        title=DASHBOARD_TITLE,
    )
app.config["suppress_callback_exceptions"] = True
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
        dcc.Store(id="local-explanation-state"),
        dcc.Store(id="global-explanation-state"),
        dcc.Store(id="data-explanation-state"),
        dcc.Store(id="prediction-explanation-state"),
        dcc.Store(id="whatif-explanation-state"),
    ]
)


class Dashboard:
    """
    The OmniXAI dashboard.

    .. code-block:: python

        dashboard = Dashboard(
            instances=instances,
            local_explanations=local_explanations,   # Set local explanation results
            global_explanations=global_explanations, # Set global explanation results
            data_explanations=None,                  # Set data analysis generated by ``DataAnalyzer``
            prediction_explanations=None,            # Set predication analysis generated by ``PredictionAnalyzer``
            class_names=class_names,                 # A list of class names
            params={"pdp": {"features": ["Age", "Education-Num", "Capital Gain",
                                         "Capital Loss", "Hours per week", "Education",
                                         "Marital Status", "Occupation"]}},
            explainer=explainer                      # Set a TabularExplainer if requiring what-if analysis.
        )
        dashboard.show()
    """

    def __init__(
        self,
        instances=None,
        local_explanations=None,
        global_explanations=None,
        data_explanations=None,
        prediction_explanations=None,
        class_names=None,
        params=None,
        explainer=None,
        second_explainer=None
    ):
        """
        :param instances: The instances to explain.
        :param local_explanations: The local explanation results.
        :param global_explanations: The global explanation results.
        :param data_explanations: The analysis of the dataset generated by ``DataAnalyzer``.
        :param prediction_explanations: The analysis of the prediction results generated by ``PredictionAnalyzer``.
        :param class_names: A list of the class names indexed by the labels, e.g.,
            ``class_name = ['dog', 'cat']`` means that label 0 corresponds to 'dog' and
            label 1 corresponds to 'cat'.
        :param params: A dict containing the additional parameters for plotting figures.
        :param explainer: A ``TabularExplainer`` explainer to enable What-if explanations for tabular tasks.
        :param second_explainer: A ``TabularExplainer`` explainer used to compare different models in What-if analysis.
        """
        if explainer is not None:
            assert isinstance(explainer, TabularExplainer), \
                "`explainer` can only be a `TabularExplainer` object."
        if second_explainer is not None:
            assert isinstance(second_explainer, TabularExplainer), \
                "`second_explainer` can only be a `TabularExplainer` object."

        board.state.set(
            instances=instances,
            local_explanations=local_explanations,
            global_explanations=global_explanations,
            data_explanations=data_explanations,
            prediction_explanations=prediction_explanations,
            class_names=class_names,
            params=params,
        )
        board.whatif_state.set(
            instances=instances,
            local_explanations=local_explanations,
            class_names=class_names,
            params=params,
            explainer=explainer,
            second_explainer=second_explainer
        )

    def show(self, host=os.getenv("HOST", "127.0.0.1"), port=os.getenv("PORT", "8050")):
        """
        Shows the dashboard.
        """
        if not _use_jupyter_dash and board.state.has_explanations():
            app.run_server(host=host, port=port, debug=False)
        elif _use_jupyter_dash and board.state.has_explanations():
            app.run_server(mode="inline", host=host, port=port, debug=False)
        else:
            warnings.warn(
                "No explanations to show are available to show in the dashboard"
            )


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def _display_page(pathname):
    return html.Div(
        id="app-container",
        children=[
            create_banner(app),
            html.Br(),
            create_layout(board.state, board.whatif_state)
        ],
    )


@app.callback(
    Output("plots", "children"),
    Input("tabs", "value"),
    [
        State("local-explanation-state", "data"),
        State("global-explanation-state", "data"),
        State("data-explanation-state", "data"),
        State("prediction-explanation-state", "data"),
        State("whatif-explanation-state", "data")
    ],
)
def _click_tab(
        tab,
        local_exp_state,
        global_exp_state,
        data_exp_state,
        prediction_exp_state,
        whatif_exp_state,
):
    if tab == "local-explanation":
        state = copy.deepcopy(board.state)
        params = json.loads(local_exp_state) \
            if local_exp_state is not None else {}
        for param, value in params.items():
            state.set_param("local", param, value)
        return create_local_explanation_layout(state)

    elif tab == "global-explanation":
        state = copy.deepcopy(board.state)
        params = json.loads(global_exp_state) \
            if global_exp_state is not None else {}
        for param, value in params.items():
            state.set_param("global", param, value)
        return create_global_explanation_layout(state)

    elif tab == "data-explanation":
        state = copy.deepcopy(board.state)
        params = json.loads(data_exp_state) \
            if data_exp_state is not None else {}
        for param, value in params.items():
            state.set_param("data", param, value)
        return create_data_explanation_layout(state)

    elif tab == "prediction-explanation":
        state = copy.deepcopy(board.state)
        params = json.loads(prediction_exp_state) \
            if prediction_exp_state is not None else {}
        for param, value in params.items():
            state.set_param("prediction", param, value)
        return create_prediction_explanation_layout(state)

    elif tab == "what-if-explanation":
        state = copy.deepcopy(board.whatif_state)
        params = json.loads(whatif_exp_state) \
            if whatif_exp_state is not None else {}
        for param, value in params.items():
            state.set_param(param, value)
        return create_what_if_layout(state)
