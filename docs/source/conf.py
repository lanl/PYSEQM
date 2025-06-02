# -- Project information -----------------------------------------------------
project = 'PYSEQM'
copyright = '2025, Los Alamos National Laboratory'
author = 'Los Alamos National Laboratory'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
]

autosummary_generate = True
autodoc_mock_imports = ["setup"]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

# Only set theme-level logo, not `html_logo`
html_theme_options = {
    "logo": {
        # "image_light": "LANL-Light.png",     
        # "image_dark": "LANL-Dark.png",     

        "image_light": "PySEQM-Light.png",
        "image_dark": "PySEQM-Light.png",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher"],  # Removed 'github-link' for compatibility
    "navigation_with_keys": False,
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc"],
    "github_url": "https://github.com/lanl/pyseqm",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/lanl/pyseqm",
            "icon": "fa-brands fa-github",
        },
    ],
}


# Important: Do NOT set html_logo here — it overrides the theme's logo
# html_logo = "_static/logo.png"  <-- REMOVE or comment this out

html_sidebars = {
    "index": [],
    "**": ["sidebar-nav-bs", "search-field", "sourcelink"],
}

html_show_sourcelink = False

