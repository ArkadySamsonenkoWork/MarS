	# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MarS'
copyright = '2026, Arkady Samsonenko, Ivan Kurgansky'
author = 'Arkady Samsonenko, Ivan Kurgansky'
release = '2026.01.11'
version = '0.0.3b1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # авто документации из docstrings
    'sphinx.ext.viewcode',  # ссылки на исходный код
    'sphinx.ext.napoleon',  # поддержка Google и NumPy стиля документации
    'sphinx.ext.todo',      # поддержка TODO
    'sphinx.ext.coverage',  # проверяет покрытие документации
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',  # условные директивы в документации
    'sphinx.ext.autosummary',
    'nbsphinx'
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,  # set to True if you want undocumented members shown
    'exclude-members': '__weakref__',

    'inherited-members': True,   # ← this shows inherited methods
    'show-inheritance': True,    # ← shows "Bases: ..." in class doc
}
autosummary_generate = True  # Generate stub pages automatically
autosummary_imported_members = True
autodoc_preserve_defaults = True

autodoc_mock_imports = ["optuna", "nevergrad", "optuna_dashboard"]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme' # тема оформления
html_static_path = ['_static']  # папка со статическими файлами (например, CSS)
todo_include_todos = True  # показывать TODO в готовой документации

napoleon_google_docstring = False
napoleon_numpy_docstring = True

napoleon_use_param = False
napoleon_use_rtype = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.insert(0, PROJECT_ROOT)


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    # Don't need to import these on RTD since they're mocked
    pass

import mars
print("mars imported from:", mars.__file__)