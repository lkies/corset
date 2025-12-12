# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Beam Corset"
copyright = "2025, Lorenz Kies"
author = "Lorenz Kies"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "groupwise"
autodoc_typehints = "both"
typehints_use_signature_return = True
typehints_defaults = "comma"

nbsphinx_execute = "always"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "text": "Beam Corset",
        "image_light": "../../misc/logo/logo_light.svg",
        "image_dark": "../../misc/logo/logo_dark.svg",
    },
}

html_favicon = "../../misc/logo/favicon.svg"


### Preprocess notebooks for RST compatibility
# TODO move this to a separate script and check timestamps to only preprocess when necessary
import json
from pathlib import Path


def process_notebook(source: Path, output: Path):
    notebook = json.loads(source.read_text(encoding="utf-8"))

    for cell in notebook.get("cells", []):
        metadata = cell.get("metadata", {})
        if cell.get("cell_type") == "markdown" and "rst" in metadata.get("tags", []):
            cell["cell_type"] = "raw"
            cell["metadata"] = metadata | {"raw_mimetype": "text/restructuredtext"}
            cell["source"] = [line.replace("\\", "\\\\") for line in cell["source"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def preprocess_notebooks(source_dir: Path, output_dir: Path):
    for source in source_dir.glob("*.ipynb"):
        output = output_dir / source.name
        process_notebook(source, output)


this_file_dir = Path(__file__).parent

preprocess_notebooks(this_file_dir / "../notebooks", this_file_dir / "gen")
