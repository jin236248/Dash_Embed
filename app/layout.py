from dash import Dash, dcc, html

layout = html.Div([
    html.H1("Word Embedding Visualizer"),
    dcc.Dropdown(
        id='word-dropdown',
        options=[],
        placeholder="Select a word"
    ),
    dcc.Graph(
        id='3d-scatter-chart'
    )
])