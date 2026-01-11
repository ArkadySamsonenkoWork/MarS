# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MarS'
copyright = '2026, Arakady Samsonenko'
author = 'Arakady Samsonenko'
release = '2026.01.11'
version = '0.0.1'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # авто документации из docstrings
    'sphinx.ext.viewcode',  # ссылки на исходный код
    'sphinx.ext.napoleon',  # поддержка Google и NumPy стиля документации
    'sphinx.ext.todo',  # поддержка TODO
    'sphinx.ext.coverage',  # проверяет покрытие документации
    'sphinx.ext.ifconfig',  # условные директивы в документации
]

templates_path = ['_templates']
exclude_patterns = []


html_theme = 'sphinx_rtd_theme' # тема оформления
html_static_path = ['_static']  # папка со статическими файлами (например, CSS)
todo_include_todos = True  # показывать TODO в готовой документации


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output




import os
import sys
sys.path.insert(0, os.path.abspath('../../'))