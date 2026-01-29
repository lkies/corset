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

## Links

- Documentation: <https://lkies.github.io/corset/intro.html>
- Source Code: <https://github.com/lkies/corset>
- Issue Tracker: <https://github.com/lkies/corset/issues>
- JupyterLite: <https://lkies.github.io/corset/jp-lite>

## Contributing

TODO

### Git Pre-Commit

```shell
git config filter.strip-notebook-output.clean 'pixi run -e dev jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
git config filter.strip-notebook-output.smudge cat
git config filter.strip-notebook-output.required true
```
