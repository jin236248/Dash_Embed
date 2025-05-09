import json

import dash
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
    # load common_word.txt
    with open("app/data/common_word.txt") as f:
        word_list = f.read().splitlines()
    return word_list

    # with open("app/data/word_list.json") as f:
    #     word_list = json.load(f)
    # return word_list


def load_model(year):
    model_path = f"app/models/sg0/{year}_e10.model"
    model = Word2Vec.load(model_path)
    return model


def get_models_and_similar_words(selected_word, years, topn):
    models, similar_words = {}, {}
    for year in years:
        models[year] = load_model(year)
        similar_words[year] = [word for word, sim in models[year].wv.most_similar(selected_word, topn=topn)]

    return models, similar_words


def get_all_embeddings(models, similar_words, selected_word, years, weight):
    all_embeddings = []

    # main word
    for year in years:
        all_embeddings.extend([models[year].wv[selected_word]] * weight)

    # similar words
    for year in years:
        for word in similar_words[year]:
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
        for word in similar_words[year]:
            if word not in all_similar_words:
                data.append(
                    {"x": reduced_embed[index][0], "y": reduced_embed[index][1], "z": reduced_embed[index][2], "word": word, "year": year, "size": 1}
                )
                all_similar_words.append(word)
                index += 1

    df = pd.DataFrame(data)
    return df


def create_fig(df, reduced_embed, selected_word, years, weight, font_size):

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="year", text="word", size="size")

    # Manually update the colors for each year
    custom_colors = {
        "1987": "red",
        "1997": "green",
        "2006": "blue",
    }
    for year, color in custom_colors.items():
        fig.for_each_trace(lambda trace: trace.update(marker=dict(color=color)) if trace.name == year else None)

    fig.update_traces(textposition="top center", textfont=dict(size=font_size))
    fig.update_layout(
        scene=dict(camera=dict(eye=dict(x=0.5, y=0.5, z=0.5))),
        hovermode=False,  # Disable hover interaction that triggers projection lines
        # font=dict(size=20),  # Adjust font size globally
    )

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
font_size = 12  # initial font size, after one callback it will be set to 25 and the bug in the font size will be fixed


def div(children):
    return html.Div(children, style={"height": "90%", "minHeight": "600px"})


def col(children):
    return dbc.Col(div(children), width=6, style={"minWidth": "600px"})


left_col = col(
    [
        html.H2("Word Embedding"),
        html.P("Some sentences"),
        dcc.Dropdown(id="word-dropdown", options=word_list, value="derivatives", clearable=False),
        dcc.Graph(id="3d-scatter-chart", style={"height": "100%"}),
    ],
)

# Add color options for the dropdowns
color_options = [
    {"label": html.Span(["Red"], style={"color": "red"}), "value": "red"},
    {"label": html.Span(["Green"], style={"color": "green"}), "value": "green"},
    {"label": html.Span(["Blue"], style={"color": "blue"}), "value": "blue"},
    {"label": html.Span(["Orange"], style={"color": "orange"}), "value": "orange"},
    {"label": html.Span(["Purple"], style={"color": "purple"}), "value": "purple"},
    {"label": html.Span(["Black"], style={"color": "black"}), "value": "black"},
]

# Add dropdowns for selecting colors for each year
color_dropdowns = dbc.Row(
    [
        dbc.Col(
            [
                html.Label("Color for 1987:"),
                dcc.Dropdown(
                    id="color-1987-dropdown",
                    options=color_options,
                    value="red",  # Default color
                    clearable=False,
                ),
            ],
        ),
        dbc.Col(
            [
                html.Label("Color for 1997:"),
                dcc.Dropdown(
                    id="color-1997-dropdown",
                    options=color_options,
                    value="green",  # Default color
                    clearable=False,
                ),
            ],
        ),
        dbc.Col(
            [
                html.Label("Color for 2006:"),
                dcc.Dropdown(
                    id="color-2006-dropdown",
                    options=color_options,
                    value="blue",  # Default color
                    clearable=False,
                ),
            ],
        ),
    ],
    className="g-0",
)

# Update the right_col to include the color dropdowns and conditional styling for the table
right_col = col(
    [
        html.H2("Most Similar Words"),
        html.P("How they change over time"),
        color_dropdowns,  # Add the color dropdowns here
        dbc.Spinner(
            [
                dash.dash_table.DataTable(
                    id="similar-words-table",
                    columns=[
                        {"name": "1987", "id": "1987"},
                        {"name": "1997", "id": "1997"},
                        {"name": "2006", "id": "2006"},
                        {"name": "1987_isunique", "id": "1987_isunique"},
                        {"name": "1997_isunique", "id": "1997_isunique"},
                        {"name": "2006_isunique", "id": "2006_isunique"},
                    ],
                    style_cell={
                        "textAlign": "center",
                        "fontFamily": "Arial",
                        "width": "33.33%",  # Make all columns equal width"
                    },
                    style_data_conditional=[],  # This will be dynamically updated in the callback
                    style_header_conditional=[
                        {"if": {"column_id": "1987_isunique"}, "display": "none"},
                        {"if": {"column_id": "1997_isunique"}, "display": "none"},
                        {"if": {"column_id": "2006_isunique"}, "display": "none"},
                    ],
                    style_cell_conditional=[
                        {"if": {"column_id": "1987_isunique"}, "display": "none"},
                        {"if": {"column_id": "1997_isunique"}, "display": "none"},
                        {"if": {"column_id": "2006_isunique"}, "display": "none"},
                    ],
                ),
            ],
            color="primary",  # Spinner color
            type="border",  # Spinner type
            fullscreen=False,  # Spinner will not cover the entire screen
        ),
    ],
)

app.layout = html.Div(
    [
        dbc.Row(
            [left_col, right_col],
            justify="center",
            style={"height": "95vh", "minHeight": "600px"},
            className="border border-1",
        ),
    ],
)


# Update the callback to include conditional styling for the table
@app.callback(
    [
        Output("3d-scatter-chart", "figure"),
        Output("similar-words-table", "data"),
        Output("similar-words-table", "style_data_conditional"),
    ],
    [
        Input("word-dropdown", "value"),
        Input("color-1987-dropdown", "value"),
        Input("color-1997-dropdown", "value"),
        Input("color-2006-dropdown", "value"),
    ],
)
def update_scatter(selected_word, color_1987, color_1997, color_2006):
    if not selected_word:
        return None, [], []

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
    fig = create_fig(df, reduced_embed, selected_word, years, weight, font_size)

    # Update colors based on dropdown selections
    custom_colors = {
        "1987": color_1987,
        "1997": color_1997,
        "2006": color_2006,
    }
    for year, color in custom_colors.items():
        fig.for_each_trace(lambda trace: trace.update(marker=dict(color=color)) if trace.name == year else None)

    # Prepare data for the table
    table_data = []

    # Preprocess the data to determine which words are unique to each column
    unique_1987 = set(similar_words["1987"]) - set(similar_words["1997"]) - set(similar_words["2006"])
    unique_1997 = set(similar_words["1997"]) - set(similar_words["1987"])
    unique_2006 = set(similar_words["2006"]) - set(similar_words["1987"])

    for i in range(topn):
        row = {
            "1987": similar_words["1987"][i],
            "1997": similar_words["1997"][i],
            "2006": similar_words["2006"][i],
            "1987_isunique": 1 if similar_words["1987"][i] in unique_1987 else 0,
            "1997_isunique": 1 if similar_words["1997"][i] in unique_1997 else 0,
            "2006_isunique": 1 if similar_words["2006"][i] in unique_2006 else 0,
        }
        table_data.append(row)
    # Prepare conditional styling for the table
    style_data_conditional = []

    # Add styles for each column based on the explicit color fields
    style_data_conditional.extend(
        [
            {
                "if": {
                    "filter_query": "{1987_isunique} > 0",
                    "column_id": "1987",
                },
                "color": color_1987,
            },
            {
                "if": {
                    "filter_query": "{1997_isunique} > 0",
                    "column_id": "1997",
                },
                "color": color_1997,
            },
            {
                "if": {
                    "filter_query": "{2006_isunique} > 0",
                    "column_id": "2006",
                },
                "color": color_2006,
            },
        ]
    )

    font_size = 25

    return fig, table_data, style_data_conditional


if __name__ == "__main__":
    app.run(debug=True)
