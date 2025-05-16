import json
import os
import re

import dash
import dash_auth
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from gensim.models import KeyedVectors, Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# helper functions


def get_years_from_models():

    def get_years(model_dir):
        year_pattern = re.compile(r"(\d{4})\.kv$")
        years = []
        for fname in os.listdir(model_dir):
            if fname.endswith(".kv"):
                match = year_pattern.match(fname)
                if match:
                    years.append(match.group(1))
                else:
                    return "Invalid file name format"
        # make sure the number of models is 3
        if len(years) != 3:
            return f"Number of models in {model_dir} is not 3"
        return sorted(years, reverse=True)

    years_sg0 = get_years("models/sg0")
    if isinstance(years_sg0, str):
        return years_sg0

    years_sg1 = get_years("models/sg1")
    if isinstance(years_sg1, str):
        return years_sg1

    if years_sg0 != years_sg1:
        return "The years in sg0 and sg1 models do not match."
    return years_sg0


def load_models(years):
    models = {}
    for sg in ["0", "1"]:
        models[sg] = {}
        for year in years:
            model_path = f"models/sg{sg}/{year}.kv"
            try:
                models[sg][year] = KeyedVectors.load(model_path)
            except Exception as e:
                return f"Error loading model for year {year} and sg {sg}: {e}"
    return models


def get_word_list(models):
    try:
        vocab = set(models["0"][years[0]].key_to_index.keys())
    except AttributeError:
        return f"Error loading vocab for year {years[0]} in sg0."

    # Check if models have same vocabulary
    for sg in ["0", "1"]:
        for year in years:
            if set(models[sg][year].key_to_index.keys()) != vocab:
                return f"Vocabulary mismatch in models for year {year} in sg{sg}."

    return sorted(list(vocab))


def create_error_modal(messages):
    return dbc.Modal(
        [
            dbc.ModalHeader("Error", close_button=True),
            dbc.ModalBody(
                [
                    html.Ul(
                        html.Li(messages[0]),
                        style={"listStyleType": "none", "padding": "0"},
                    )
                ]
            ),
        ],
        id="error-modal",
        is_open=True,
    )


def get_similar_words(selected_word, models, years, sg, topn):
    similar_words = {}
    for year, key in zip(years, keys):
        similar_words[key] = [word for word, sim in models[sg][year].most_similar(selected_word, topn=topn)]
    return similar_words


def get_all_embeddings(models, similar_words, selected_word, years, weight):
    # models is already [sg] in this function
    all_embeddings = []

    # main word
    for year in years:
        all_embeddings.extend([models[year][selected_word]] * weight)

    # similar words
    for year, key in zip(years, keys):
        for word in similar_words[key]:
            all_embeddings.append(models[year][word])

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
    for year, key in zip(years, keys):
        for word in similar_words[key]:
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
server = app.server

USERNAME_PASSWORD_PAIRS = [["digitalscholarship", "2025"]]

auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

# Layout of the app
font_size = 12  # initial font size, after one callback it will be set to 25 and the bug in the font size will be fixed
text_font_size = 14
text_font = "Verdana"

# Error messages
error_messages = []

# Get years from models
years = get_years_from_models()
keys = ["3", "2", "1"]
if isinstance(years, str):
    error_messages.append(years)
    years = ["2006", "1997", "1987"]  # Default years if the model loading fails

# Load models
models = load_models(years)
if isinstance(models, str):
    error_messages.append(models)
    models = {}  # Default models if the model loading fails
# Check if all models have same vocabulary
word_list = []
if len(models) > 0:
    word_list = get_word_list(models)
    if isinstance(word_list, str):
        error_messages.append(word_list)
        word_list = []

# Set default word
default_word = None
if word_list:
    default_word = word_list[0] if "derivatives" not in word_list else "derivatives"


def div(children, bg_color):
    return html.Div(
        children,
        style={
            "height": "100%",
            "minHeight": "450px",
            "backgroundColor": bg_color,
            "borderRadius": "15px",
            "boxShadow": "0 8px 16px rgba(0, 0, 0, 0.4)",
            "display": "flex",  # Use flexbox for the container
            "flexDirection": "column",  # Ensure vertical layout
        },
        className="m-2",  # border border-danger",
    )


def col(children, bg_color="#002147"):
    return dbc.Col(
        div(children, bg_color),
        width=6,
        style={"minWidth": "600px"},
    )


def upper_box(children):
    return html.Div(children, style={"height": "125px"}, className="m-2")  # border border-light",


def flex_div(children):
    return html.Div(
        children,
        style={
            "flex": "1",
            "borderRadius": "15px",
            "backgroundColor": "white",
            "display": "flex",  # Use flexbox for the graph container
            "flexDirection": "column",  # Ensure vertical layout
            "justifyContent": "center",  # Center the content vertically
            # "alignItems": "center",  # Center the content horizontally
        },
        className="m-2",  # border border-success",
    )


def lower_box(children):
    return html.Div(children, style={"height": "25px"}, className="m-2 ms-3 me-3")  # border border-light",


dropdown_style = {
    "borderRadius": "30px",
    "fontSize": text_font_size,
    "fontFamily": text_font,
}

word_dropdown = dbc.Row(
    [
        dbc.Col(
            html.Label(
                "Word:",
                style={
                    "fontSize": text_font_size,
                    "fontFamily": text_font,
                    "marginRight": "0px",  # Space between label and dropdown
                },
            ),
            width="auto",
        ),
        dbc.Col(
            dcc.Dropdown(
                id="word-dropdown",
                options=word_list,
                value=default_word,
                clearable=False,
                style=dropdown_style,
            ),
            width=6,
        ),
    ],
    justify="center",
    align="center",  # Align label and dropdown vertically
    className="mt-4 mb-2",
)

sg_radio = dbc.Row(
    dbc.Col(
        dcc.RadioItems(
            id="sg-radio",
            options=[
                {"label": "Continuous Bag of Words (CBOW)", "value": "0"},
                {"label": "Skip-Gram (SG)", "value": "1"},
            ],
            value="0",  # Default to CBOW
            inline=True,  # Display options horizontally
            style={"fontSize": text_font_size, "fontFamily": text_font},
            inputStyle={"marginRight": "10px"},  # Add space between the radio button and label
            labelStyle={"marginRight": "20px"},  # Add space between the two choices
        ),
        width="auto",
    ),
    justify="center",
    className="mb-2",
)

# Add color options for the dropdowns
colors = [
    ("#ea5545", "Red"),
    ("#f46a9b", "Pink"),
    ("#ef9b20", "Orange"),
    ("#edbf33", "Amber"),
    ("#ede15b", "Yellow"),
    ("#bdcf32", "Olive"),
    ("#87bc45", "Green"),
    ("#27aeef", "Blue"),
    ("#b33dc6", "Purple"),
]
color_options = [{"label": html.Span([c[1]], style={"color": c[0]}), "value": c[0]} for c in colors]

color_dropdowns = dbc.Row(
    [
        dbc.Col(
            [
                dcc.Dropdown(
                    id="color-1-dropdown",
                    options=color_options,
                    value="#ea5545",  # Default color
                    clearable=False,
                    style=dropdown_style,
                ),
            ],
        ),
        dbc.Col(
            [
                dcc.Dropdown(
                    id="color-2-dropdown",
                    options=color_options,
                    value="#87bc45",  # Default color
                    clearable=False,
                    style={"borderRadius": "30px", "fontSize": text_font_size, "fontFamily": text_font},
                ),
            ],
        ),
        dbc.Col(
            [
                dcc.Dropdown(
                    id="color-3-dropdown",
                    options=color_options,
                    value="#27aeef",  # Default color
                    clearable=False,
                    style={"borderRadius": "30px", "fontSize": text_font_size, "fontFamily": text_font},
                ),
            ],
        ),
    ],
    className="m-2 g-2",
)

graph = dcc.Graph(
    id="3d-scatter-chart",
    style={"height": "90%"},
    className="m-2",
)

left_col = col(
    [
        upper_box(
            [
                html.H2(
                    "Word Embeddings",
                    style={
                        "color": "#002147",
                        "textAlign": "center",
                        "fontWeight": 900,  # Increased bold weight
                    },
                    className="mt-2",
                ),
                html.P(
                    [
                        "Visualize the evolution of the selected word and its closest words.",
                        html.Br(),
                        "Drag to rotate the chart and scroll to zoom.",
                    ],
                    style={"color": "#002147", "textAlign": "center", "fontSize": text_font_size, "fontFamily": text_font},
                ),
            ]
        ),
        flex_div(graph),
        lower_box(
            html.P(
                [
                    "The embeddings are trained from the New York Times Annotated Corpus.",
                ],
                style={"color": "#002147", "textAlign": "left", "fontSize": text_font_size, "fontFamily": text_font},
            )
        ),
    ],
    bg_color="#B9D6F2",
)


data_table = dash.dash_table.DataTable(
    id="similar-words-table",
    columns=[
        {"name": years[2], "id": "1"},
        {"name": years[1], "id": "2"},
        {"name": years[0], "id": "3"},
        {"name": "1_isunique", "id": "1_isunique"},
        {"name": "2_isunique", "id": "2_isunique"},
        {"name": "3_isunique", "id": "3_isunique"},
    ],
    style_table={
        "border": "1px solid #B0B0B0",  # Grey border
        "borderRadius": "20px",  # Adjust for rounder effect
        "overflow": "hidden",  # Prevents clipped corners if needed
        "margin": "8px",
        "width": "97%",
    },
    style_cell={
        "textAlign": "center",
        "fontFamily": text_font,
        "fontSize": text_font_size,
        "padding": "1px",  # Increase padding to increase row height
        "width": "33.33%",  # Make all columns equal width"
    },
    style_header={
        "fontWeight": "bold",  # Make the header bold
        "textAlign": "center",
        "fontFamily": text_font,
        "fontSize": text_font_size,
        "backgroundColor": "#f8f9fa",  # Optional: Add a light background color for the header
    },
    style_data_conditional=[],  # This will be dynamically updated in the callback
    style_header_conditional=[
        {"if": {"column_id": "1_isunique"}, "display": "none"},
        {"if": {"column_id": "2_isunique"}, "display": "none"},
        {"if": {"column_id": "3_isunique"}, "display": "none"},
    ],
    style_cell_conditional=[
        {"if": {"column_id": "1_isunique"}, "display": "none"},
        {"if": {"column_id": "2_isunique"}, "display": "none"},
        {"if": {"column_id": "3_isunique"}, "display": "none"},
    ],
)
# Update the right_col to include the color dropdowns and conditional styling for the table
right_col = col(
    [
        upper_box(
            [
                html.H2(
                    "Most Similar Words",
                    style={
                        "color": "white",
                        "textAlign": "center",
                        "fontWeight": 900,  # Increased bold weight
                    },
                    className="mt-2",
                ),
                html.P(
                    [
                        # "Select a word and model, then adjust the color using the dropdowns.",
                        # html.Br(),
                        "Try: ",
                        html.B("derivatives"),
                        ", ",
                        html.B("bubble"),
                        ", ",
                        html.B("underwriting"),
                        ", ",
                        html.B("leveraged"),
                        ", ",
                        html.B("networking"),
                        ", ",
                        html.B("hybrid"),
                        ", ",
                        html.B("globalization"),
                        ", ",
                        html.B("nasdaq"),
                        ", and ",
                        html.B("options"),
                        ". ",
                        "They highlight social and economic shifts over the two decades.",
                    ],
                    style={"color": "white", "textAlign": "center", "fontSize": text_font_size, "fontFamily": text_font},
                ),
            ]
        ),
        flex_div([word_dropdown, sg_radio]),
        color_dropdowns,  # Add the color dropdowns here
        dbc.Spinner(data_table, color="light"),
        lower_box(
            html.P(
                [
                    "Source code",
                    html.A(
                        html.Img(
                            src="/assets/github_white.png",
                            alt="GitHub",
                            style={"height": "18px", "verticalAlign": "middle", "marginRight": "4px", "marginLeft": "8px"},
                        ),
                        href="https://github.com/jin236248/Dash_Embed",
                        target="_blank",
                        style={"textDecoration": "none"},
                    ),
                    html.A(
                        "Dash_Embed",
                        href="https://github.com/jin236248/Dash_Embed",
                        target="_blank",
                        style={
                            "color": "white",
                            "fontWeight": "bold",
                            "textDecoration": "underline",
                            "marginLeft": "4px",
                        },
                    ),
                ],
                style={"color": "white", "textAlign": "right", "fontSize": text_font_size, "fontFamily": text_font},
            )
        ),
    ],
)

app.layout = html.Div(
    [
        dbc.Row(
            [left_col, right_col],
            justify="center",
            style={
                "height": "90vh",
                "minHeight": "600px",
            },
            className="g-2",
        ),
        html.Div(id="error-modal-container"),  # Placeholder for error modal
    ],
)


# Update the callback to include conditional styling for the table
@app.callback(
    [
        Output("3d-scatter-chart", "figure"),
        Output("similar-words-table", "data"),
        Output("similar-words-table", "style_data_conditional"),
        Output("error-modal-container", "children"),  # Output for the error modal
    ],
    [
        Input("word-dropdown", "value"),
        Input("sg-radio", "value"),  # Add the radio button input
        Input("color-1-dropdown", "value"),
        Input("color-2-dropdown", "value"),
        Input("color-3-dropdown", "value"),
    ],
)
def update_scatter(selected_word, sg, color_1, color_2, color_3):
    # check errors
    if error_messages:
        return None, [], [], create_error_modal(error_messages)

    if not selected_word:
        return None, [], [], no_update

    global font_size, models

    # parameters
    weight = 5
    topn = 15

    # get models and similar words
    similar_words = get_similar_words(selected_word, models, years, sg, topn)

    # list all words and their embeddings
    all_embeddings = get_all_embeddings(models[sg], similar_words, selected_word, years, weight)

    # Reduce dimensions to 3
    reduced_embed = reduce_embed(all_embeddings)

    # Prepare data for plotting
    df = prepare_df(selected_word, reduced_embed, similar_words, years, weight)

    # Create 3D scatter plot
    fig = create_fig(df, reduced_embed, selected_word, years, weight, font_size)

    # Update colors based on dropdown selections
    custom_colors = {
        years[2]: color_1,
        years[1]: color_2,
        years[0]: color_3,
    }
    for year, color in custom_colors.items():
        fig.for_each_trace(lambda trace: trace.update(marker=dict(color=color)) if trace.name == year else None)

    # Prepare data for the table
    table_data = []

    # Preprocess the data to determine which words are unique to each column
    unique_1 = set(similar_words["1"]) - set(similar_words["2"]) - set(similar_words["3"])
    unique_2 = set(similar_words["2"]) - set(similar_words["1"]) - set(similar_words["3"])
    unique_3 = set(similar_words["3"]) - set(similar_words["1"]) - set(similar_words["2"])
    for i in range(topn):
        row = {
            "1": similar_words["1"][i],
            "2": similar_words["2"][i],
            "3": similar_words["3"][i],
            "1_isunique": 1 if similar_words["1"][i] in unique_1 else 0,
            "2_isunique": 1 if similar_words["2"][i] in unique_2 else 0,
            "3_isunique": 1 if similar_words["3"][i] in unique_3 else 0,
        }
        table_data.append(row)
    # Prepare conditional styling for the table
    style_data_conditional = []

    # Add styles for each column based on the explicit color fields
    style_data_conditional.extend(
        [
            {
                "if": {
                    "filter_query": "{1_isunique} > 0",
                    "column_id": "1",
                },
                "color": color_1,
            },
            {
                "if": {
                    "filter_query": "{2_isunique} > 0",
                    "column_id": "2",
                },
                "color": color_2,
            },
            {
                "if": {
                    "filter_query": "{3_isunique} > 0",
                    "column_id": "3",
                },
                "color": color_3,
            },
        ]
    )

    font_size = 25

    return fig, table_data, style_data_conditional, no_update


if __name__ == "__main__":
    app.run(debug=False)
