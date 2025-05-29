# Dash_Embed

A Dash web application for visualizing the evolution of word embeddings and their most similar words over time.

## Features

- **3D Scatter Plot**: Visualize the trajectory of a selected word and its most similar words across multiple years.
- **Interactive Table**: Compare the top similar words for each year and highlight unique words per year.
- **Customizable Colors**: Change the color scheme for each year using dropdowns.
- **Model Selection**: Switch between Continuous Bag of Words (CBOW) and Skip-Gram (SG) models.

## Getting Started

### Prerequisites

- Python 3.11
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/jin236248/Dash_Embed.git
    cd Dash_Embed
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare models and data:**

    - **Models:**  
      You need word embedding models in Gensim's `.kv` format.  
      Place 3 files in both `models/sg0/` and `models/sg1/` subfolders.  
      The filenames must be the year, and the years must match between both folders.  
      If you only have one version of the model, copy the same `.kv` file(s) into both `sg0` and `sg1`.

      ```
      models/
        sg0/
          1987.kv
          1997.kv
          2006.kv
        sg1/
          1987.kv
          1997.kv
          2006.kv
      ```

    - **How to train a model with Gensim:**  
      See the [Gensim Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html) for how to train a model:
      ```python
      from gensim.models import Word2Vec
      model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
      model.wv.save("1987.kv")
      ```

    - **How to convert other formats to KeyedVectors format:**  
      If your embeddings are in another format (such as word2vec `.bin` or `.txt`), you can convert them to Gensim's `.kv` format.  
      See the [Gensim KeyedVectors documentation](https://radimrehurek.com/gensim/models/keyedvectors.html#loading-and-saving-vectors) for details:
      ```python
      from gensim.models import KeyedVectors
      kv = KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
      kv.save('1987.kv')
      ```

      More info:  
      - [Loading and saving vectors in Gensim](https://radimrehurek.com/gensim/models/keyedvectors.html#loading-and-saving-vectors)
      - [Gensim Tutorials](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)

### Running the App

```bash
python app.py
```

The app will be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/).

## Acknowledgements

- Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/).