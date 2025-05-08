from dash import dcc, html, Input, Output
import dash
import json
import numpy as np
import plotly.express as px

# Load word embeddings from JSON files
def load_embeddings(year):
    with open(f'app/data/embeddings_{year}.json') as f:
        return json.load(f)

# Load word list
with open('app/data/word_list.json') as f:
    word_list = json.load(f)

# Initialize embeddings for different years
embeddings_1987 = load_embeddings(1987)
embeddings_1997 = load_embeddings(1997)
embeddings_2006 = load_embeddings(2006)

# Create a Dash callback to update the 3D scatter plot
def register_callbacks(app):
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('word-dropdown', 'value')
    )
    def update_scatter(selected_word):
        if selected_word is None:
            return px.scatter_3d()

        # Get embeddings for the selected word and its similar words
        similar_words_1987 = embeddings_1987.get(selected_word, [])
        similar_words_1997 = embeddings_1997.get(selected_word, [])
        similar_words_2006 = embeddings_2006.get(selected_word, [])

        # Prepare data for plotting
        data = []
        for year, similar_words in zip([1987, 1997, 2006], 
                                        [similar_words_1987, similar_words_1997, similar_words_2006]):
            for word, embedding in similar_words:
                data.append({
                    'word': word,
                    'x': embedding[0],
                    'y': embedding[1],
                    'z': embedding[2],
                    'year': year
                })

        df = pd.DataFrame(data)

        # Create 3D scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='year', text='word',
                            title=f'Similar Words for "{selected_word}"',
                            labels={'year': 'Year'},
                            color_continuous_scale=px.colors.sequential.Viridis)

        return fig