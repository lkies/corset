# Beam Corset

Beam Corset is a Gaussian optics mode matching tool made for use in Jupyter notebooks.

## Key Features

- Lens placement in multiple shifting regions
- Ensure minimal distances between lenses
- Constrain beam radius to ensure the beam fits through apertures
- Account for existing fixed lenses
- Detailed reachability and sensitivity analysis of solutions

## Installation

Install from PyPI:

```shell
pip install beam-corset
```

> [!Tip]
Try Beam Corset in your browser with [JupyterLite](https://lkies.github.io/corset/jp-lite/lab/index.html?path=template.ipynb)!

## Links

- Documentation: <https://lkies.github.io/corset>
- Source Code: <https://github.com/lkies/corset>
- Issue Tracker: <https://github.com/lkies/corset/issues>
- JupyterLite: <https://lkies.github.io/corset/jp-lite>

## Information for Developers

This project is managed and built using [Pixi](https://pixi.prefix.dev/latest/installation/), see their documentation for more information on dependency management and other features. To install the development environment for usage in Jupyter notebooks, run:

```shell
pixi install -e dev
```

The [`pyproject.toml`](./pyproject.toml) file defines the following tasks:

- `build`: Build the package
- `publish`: Publish the package to PyPI
- `build-docs`: Build the documentation
- `build-jp-lite`: Build the JupyterLite instance for the documentation (does not work properly on Windows)
- `pages`: Executes `build-docs` and `build-jp-lite` to build the web pages for GitHub Pages

Tasks can be executed with:

```shell
pixi run [task]
```

They will automatically be executed in their correct Python environment. Note that this only works if no environment has been activated with `pixi shell -e [env]`.

To prevent committing notebook outputs to the repository and producing unnecessary diffs, set up the appropriate filters with the following shell commands.

```shell
git config filter.strip-notebook-output.clean 'pixi run -e dev jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config filter.strip-notebook-output.smudge cat
git config filter.strip-notebook-output.required true
```
