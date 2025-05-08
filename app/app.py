import json

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# helper functions


# Load word list
def load_word_list():
    with open("app/data/word_list.json") as f:
        word_list = json.load(f)
    return word_list


def load_model(year):
    model_path = f"app/models/sg0/{year}_e10.model"
    model = Word2Vec.load(model_path)
    return model


def get_models_and_similar_words(selected_word, years, topn):
    models, similar_words = {}, {}
    for year in years:
        models[year] = load_model(year)
        similar_words[year] = models[year].wv.most_similar(selected_word, topn=topn)
    return models, similar_words


def get_all_embeddings(models, similar_words, selected_word, years, weight):
    all_embeddings = []

    # main word
    for year in years:
        all_embeddings.extend([models[year].wv[selected_word]] * weight)

    # similar words
    for year in years:
        for word, similarity in similar_words[year]:
            all_embeddings.append(models[year].wv[word])

    return all_embeddings


def reduce_embed(all_embeddings):
    embeddings = np.array(all_embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=3)
    reduced_embed_prescaled = pca.fit_transform(embeddings_scaled)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    reduced_embed = scaler.fit_transform(reduced_embed_prescaled)
    return reduced_embed


def prepare_df(selected_word, reduced_embed, similar_words, years, weight):
    data = []
    index = 0

    # Add the selected word to the plot
    for year in years:
        data.append(
            {"x": reduced_embed[index][0], "y": reduced_embed[index][1], "z": reduced_embed[index][2], "word": selected_word, "year": year, "size": 5}
        )
        index += weight

    # Add similar words to the plot
    all_similar_words = []
    for year in years:
        for word, similarity in similar_words[year]:
            if word not in all_similar_words:
                data.append(
                    {"x": reduced_embed[index][0], "y": reduced_embed[index][1], "z": reduced_embed[index][2], "word": word, "year": year, "size": 2}
                )
                index += 1

    df = pd.DataFrame(data)
    return df


def create_fig(df, reduced_embed, selected_word, years, weight):
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="year", text="word", size="size")
    fig.update_traces(textposition="top center")

    # Add grey lines connecting the three main words

    # connect first and second, and second and third
    for idx in [0, weight]:
        start_coords = reduced_embed[idx]  # Coordinates of the first main word
        end_coords = reduced_embed[idx + weight]  # Coordinates of the other main word
        fig.add_trace(
            go.Scatter3d(
                x=[start_coords[0], end_coords[0]],
                y=[start_coords[1], end_coords[1]],
                z=[start_coords[2], end_coords[2]],
                mode="lines",
                line=dict(color="grey", width=1, dash="dash"),
                showlegend=False,
            )
        )

    return fig


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
word_list = load_word_list()
app.layout = html.Div(
    [
        dcc.Dropdown(id="word-dropdown", options=[{"label": word, "value": word} for word in word_list], value=word_list[0], clearable=False),
        dcc.Graph(id="3d-scatter-chart"),
    ]
)


# Callback to update the 3D scatter chart
@app.callback(Output("3d-scatter-chart", "figure"), [Input("word-dropdown", "value")])
def update_scatter(selected_word):

    # parameters
    years = ["2006", "1997", "1987"]
    weight = 2
    topn = 3

    # get models and similar words
    models, similar_words = get_models_and_similar_words(selected_word, years, topn)

    # list all words and their embeddings
    all_embeddings = get_all_embeddings(models, similar_words, selected_word, years, weight)

    # Reduce dimensions to 3
    reduced_embed = reduce_embed(all_embeddings)

    # Prepare data for plotting
    df = prepare_df(selected_word, reduced_embed, similar_words, years, weight)

    # Create 3D scatter plot
    fig = create_fig(df, reduced_embed, selected_word, years, weight)

    return fig


if __name__ == "__main__":
    app.run(debug=True)
