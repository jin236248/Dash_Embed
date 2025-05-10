# Dash_Embed

A Dash web application for visualizing the evolution of word embeddings and their most similar words over time, trained on the New York Times corpus.

## Features

- **3D Scatter Plot**: Visualize the trajectory of a selected word and its most similar words across two decades (1987, 1997, 2006).
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

### Running the App

```bash
python app.py
```

The app will be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/).


## Project Structure

```
Dash_Embed/
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── assets/
│   └── github.png
├── models/
│   ├── sg0/
│   └── sg1/
├── data/
│   └── common_word.txt
```

## Acknowledgements

- Embeddings trained from the New York Times corpus.
- Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/).