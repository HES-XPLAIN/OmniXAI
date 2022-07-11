#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from dash import dcc
from dash import html
from .utils import create_explanation_layout


def create_control_panel(state) -> html.Div:
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.P("Plots"),
            html.Div(
                id="select-plots-parent-prediction",
                children=[
                    dcc.Dropdown(
                        id="select-plots-prediction",
                        options=[{"label": s, "value": s} for s in state.get_plots("prediction")],
                        value=state.get_display_plots("prediction"),
                        multi=True,
                        style={"width": "350px"},
                    )
                ],
            ),
            html.Br(),
            html.P("Number of figures per row"),
            dcc.Dropdown(
                id="select-num-figures-prediction",
                options=[{"label": "1", "value": "1"}, {"label": "2", "value": "2"}],
                value=str(state.get_num_figures_per_row("prediction")),
                style={"width": "350px"},
            ),
        ],
    )


def create_right_column(state) -> html.Div:
    explanation_views = create_explanation_layout(state, explanation_type="prediction")
    return html.Div(
        id="right-column-prediction",
        children=explanation_views
    )


def create_prediction_explanation_layout(state) -> html.Div:
    return html.Div(
        id="prediction_explanation_views",
        children=[
            # Left column
            html.Div(
                id="left-column-prediction",
                className="three columns",
                children=[
                    create_control_panel(state)
                ],
            ),
            # Right column
            html.Div(
                className="nine columns",
                children=create_right_column(state)
            )
        ]
    )
