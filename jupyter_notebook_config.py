# Configuration file for Jupyter Notebook.

c = get_config()  # noqa

# Do not open a browser, since we are on a server without GUI.
c.NotebookApp.open_browser = False

# Disable authentication, since the server liste only local connections.
c.NotebookApp.password = ""
c.NotebookApp.token = ""

c.NotebookApp.notebook_dir = "notebooks"
