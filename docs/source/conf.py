# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PYSEQM'
copyright = '2025, Los Alamos National Laboratory'
author = 'Los Alamos National Laboratory'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',       # Support for Google/NumPy docstring formats
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.autosummary',    # Generate summary tables
    'sphinx_copybutton',
]

autosummary_generate = True  # Automatically create summary tables
autodoc_mock_imports = ["setup"]



templates_path = ['_templates']
exclude_patterns = []

html_theme = "pydata_sphinx_theme"

html_logo = "_static/LANL_logo.png"  # Still needed for some display behavior

html_theme_options = {
    "logo": {
        "image_light": "LANL_logo.png",
        "image_dark": "LANL_logo.png",
    },
    "navbar_start": ["navbar-logo"],
    "navigation_depth": 3,
    "show_prev_next": False,
    "navigation_with_keys": False,
    "sidebar_hide_name": True,
}

html_show_sourcelink = False


html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [ "theme-switcher"],
    "navbar_links": [],  # deprecated but still parsed by older builds
    "header_links_before_dropdown": 5,
    "navigation_with_keys": False,
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc"],
    "external_links": [],
}

html_sidebars = {
    "index": [],  # Hide all sidebars on the landing page
    "**": ["sidebar-nav-bs", "search-field", "sourcelink"],  # Default for all other pages
}




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

