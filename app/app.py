import json

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


# Load word embeddings
def load_embeddings(year):
    with open(f"app/data/embeddings_{year}.json") as f:
        return json.load(f)


# Load word list
with open("app/data/word_list.json") as f:
    word_list = json.load(f)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = html.Div(
    [
        dcc.Dropdown(id="word-dropdown", options=[{"label": word, "value": word} for word in word_list], value=word_list[0], clearable=False),
        dcc.Graph(id="3d-scatter-chart"),
    ]
)


# Callback to update the 3D scatter chart
@app.callback(Output("3d-scatter-chart", "figure"), [Input("word-dropdown", "value")])
def update_scatter(selected_word):
    # Load embeddings for each year
    embeddings_1987 = load_embeddings(1987)
    embeddings_1997 = load_embeddings(1997)
    embeddings_2006 = load_embeddings(2006)

    # Get the selected word's embedding and similar words
    selected_embedding = embeddings_2006[selected_word]
    similar_words = [word for word in embeddings_2006 if word != selected_word]

    # Prepare data for plotting
    data = []
    for year, embeddings in zip([1987, 1997, 2006], [embeddings_1987, embeddings_1997, embeddings_2006]):
        for word in similar_words:
            if word in embeddings:
                data.append({"x": embeddings[word][0], "y": embeddings[word][1], "z": embeddings[word][2], "word": word, "year": year})

    df = pd.DataFrame(data)

    # Create 3D scatter plot
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="year", text="word")
    fig.update_traces(textposition="top center")

    return fig


if __name__ == "__main__":
    app.run(debug=True)
