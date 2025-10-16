# Configuration file for Jupyter Notebook.

c = get_config()  # noqa

# Do not open a browser, since we are on a server without GUI.
c.ServerApp.open_browser = False

# Disable authentication (empty password/token).
c.IdentityProvider.token = ""
c.ServerApp.password = ""

# Set the root directory for notebooks.
c.ServerApp.root_dir = "notebooks"

# Remove input cells from the output.
c.HTMLExporter.exclude_input = True
