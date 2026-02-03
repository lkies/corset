.. Beam Corset documentation master file, created by
   sphinx-quickstart on Wed Dec  3 22:35:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

Beam Corset is a Gaussian optics mode matching tool made for use in Jupyter notebooks.

**Key Features:**

- Lens placement in multiple shifting regions
- Ensure minimal distances between lenses
- Constrain beam radius to ensure the beam fits through apertures
- Account for existing fixed lenses
- **Detailed reachability and sensitivity analysis of solutions**


Installation
------------

Install from PyPI:

.. code-block:: shell
   
   pip install beam-corset

..  tip::
   Try Beam Corset in your browser with `JupyterLite <https://lkies.github.io/corset/jp-lite/lab/index.html?path=template.ipynb>`_!


Links
-----

- Documentation: https://lkies.github.io/corset
- Source Code: https://github.com/lkies/corset
- Issue Tracker: https://github.com/lkies/corset/issues
- JupyterLite: https://lkies.github.io/corset/jp-lite


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   gen/basic_usage.ipynb
   gen/lenses_and_lens_lists.ipynb
   gen/shifting_ranges_and_selections.ipynb
   gen/analyzing_solutions.ipynb
   gen/beam_constraints.ipynb
   gen/fixed_optics.ipynb
   gen/configuration.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   analysis
   config
   core
   database
   plot
   serialize
   solver

