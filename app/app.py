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
            {"x": reduced_embed[index][0], "y": reduced_embed[index][1], "z": reduced_embed[index][2], "word": selected_word, "year": year, "size": 3}
        )
        index += weight

    # Add similar words to the plot
    all_similar_words = []
    for year in years:
        for word, similarity in similar_words[year]:
            if word not in all_similar_words:
                data.append(
                    {"x": reduced_embed[index][0], "y": reduced_embed[index][1], "z": reduced_embed[index][2], "word": word, "year": year, "size": 1}
                )
                all_similar_words.append(word)
                index += 1

    df = pd.DataFrame(data)
    return df


def create_fig(df, reduced_embed, selected_word, years, weight):
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="year", text="word", size="size")
    # Disable hover on dots textfont=dict(size=10) hoverinfo="skip", hovertemplate=None
    fig.update_traces(textposition="top center", textfont=dict(size=10))
    # Disable projected lines when hovering
    fig.update_layout(
        scene=dict(
            # xaxis=dict(showspikes=False),  # Disable projection lines on X-axis
            # yaxis=dict(showspikes=False),  # Disable projection lines on Y-axis
            # zaxis=dict(showspikes=False),  # Disable projection lines on Z-axis
            camera=dict(eye=dict(x=0.5, y=0.5, z=0.5)),
        ),
        hovermode=False,  # Disable hover interaction that triggers projection lines
        font=dict(size=10),  # Adjust font size globally
    )

    # Remove grey plane by disabling the zero planes and setting background to white
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(showbackground=False, tickvals=[-1, 1], showgrid=True, gridcolor="black"),  # Remove background plane for X-axis
    #         yaxis=dict(showbackground=False, tickvals=[-1, 1], showgrid=True, gridcolor="black"),  # Remove background plane for Y-axis
    #         zaxis=dict(showbackground=False, tickvals=[-1, 1], showgrid=True, gridcolor="black"),  # Remove background plane for Z-axis
    #     )
    # )

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
                line=dict(color="black", width=2),  #  dash="longdash"
                showlegend=False,
            )
        )
        # Add the arrowhead using a cone
        direction = np.array(start_coords) - np.array(end_coords)
        fig.add_trace(
            go.Cone(
                x=[start_coords[0]],
                y=[start_coords[1]],
                z=[start_coords[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                sizemode="absolute",
                sizeref=0.05,  # Adjust the size of the arrowhead
                anchor="tip",
                colorscale=[[0, "black"], [1, "black"]],
                showscale=False,
            )
        )

    return fig


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
word_list = load_word_list()
font_size = 10
app.layout = html.Div(
    [
        dcc.Dropdown(id="word-dropdown", options=word_list, clearable=False),
        # html.Div(id="output-container", style={"height": "80vh"}),
        dcc.Graph(id="3d-scatter-chart", style={"height": "80vh"}),
    ]
)


# Callback to update the 3D scatter chart
@app.callback(Output("3d-scatter-chart", "figure"), [Input("word-dropdown", "value")])
def update_scatter(selected_word):
    if not selected_word:
        return None

    global font_size

    # parameters
    years = ["2006", "1997", "1987"]
    weight = 5
    topn = 15

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

    # update font size otherwise it will get smaller the second time
    # font_size = 20

    return fig


if __name__ == "__main__":
    app.run(debug=True)
