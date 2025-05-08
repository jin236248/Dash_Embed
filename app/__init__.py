from dash import Dash

app = Dash(__name__)

from . import layout
from . import callbacks

app.layout = layout.layout

if __name__ == '__main__':
    app.run_server(debug=True)