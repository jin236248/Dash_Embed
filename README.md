# Dash Word Embedding Project

This project is a Plotly Dash application that allows users to explore word embeddings from different years (1987, 1997, and 2006) through a 3D scatter chart. Users can select a word from a predefined list, and the application will display the selected word along with its most similar words, highlighting differences in embeddings across the specified years.

## Project Structure

```
dash-word-embedding
├── app
│   ├── __init__.py
│   ├── app.py
│   ├── callbacks.py
│   ├── layout.py
│   ├── assets
│   │   └── styles.css
│   └── data
│       ├── embeddings_1987.json
│       ├── embeddings_1997.json
│       ├── embeddings_2006.json
│       └── word_list.json
├── requirements.txt
├── runtime.txt
├── Procfile
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dash-word-embedding
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app/app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:8050` to view the application.

3. Select a word from the dropdown menu to visualize its embedding and the embeddings of its most similar words from the years 1987, 1997, and 2006.

## Deployment

To deploy the application on Heroku:

1. Ensure you have the Heroku CLI installed and are logged in.
2. Create a new Heroku app:
   ```
   heroku create <app-name>
   ```

3. Push the code to Heroku:
   ```
   git push heroku main
   ```

4. Open the app in your browser:
   ```
   heroku open
   ```

## Acknowledgments

This project utilizes Plotly Dash for interactive web applications and relies on pre-trained word embeddings for visualization.