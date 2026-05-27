# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import inspect

sys.path.insert(0, os.path.abspath('../../src'))

project = 'Neuro-Fuzzy Toolbox'
copyright = '2025, Juan Suárez'
author = 'Juan Suárez'
release = '0.0.1'

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon']

latex_engine = "xelatex"   # mejor soporte unicode

latex_elements = {
    # Tamaño de papel y cuerpo
    "papersize": "a4paper",
    "pointsize": "11pt",

    # Envuelve líneas largas en bloques de código e inline-literals
    "sphinxsetup": (
        "verbatimwrapslines=true, "   # envuelve líneas en code-block
        "verbatimforcewraps=true, "   # fuerza quiebre si aún no alcanza
        "inlineliteralwraps=true"     # permite quiebres en ``inline code``
    ),

    # Ajustes LaTeX de bajo nivel
    "preamble": r"""
\setlength{\emergencystretch}{3em} % ayuda contra overfull \hbox
\setlength{\headheight}{14pt}      % elimina el warning de fancyhdr
\providecommand{\tightlist}{}      % evita avisos con listas "compactas"

% Hace \newline más tolerante si aparece "fuera de línea"
\let\orignewline\newline
\renewcommand{\newline}{\leavevmode\orignewline}
""",

    # --- OPCIONAL ---
    # Si NO quieres instalar FreeSerif en el sistema,
    # descomenta las 3 líneas siguientes para usar DejaVu:
    # "fontpkg": r"""
    # \setmainfont{DejaVu Serif}
    # \setsansfont{DejaVu Sans}
    # \setmonofont{DejaVu Sans Mono}
    # """,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False 
napoleon_include_init_with_doc = True  # Incluye docstrings de __init__

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 4,
}

html_static_path = ['_static']

def skip_pytorch_methods(app, what, name, obj, skip, options):
    """
    Omite miembros cuyo código fuente se encuentre en la instalación de PyTorch.
    """
    try:
        source_file = inspect.getsourcefile(obj)
    except Exception:
        source_file = None

    if source_file:
        source_file = os.path.abspath(source_file)
        if "site-packages" in source_file and "torch" in source_file:
            return True

    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_pytorch_methods)
