# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scTimeBench'
copyright = '2026, Adrien Osakwe & Eric H. Huang'
author = 'Adrien Osakwe & Eric H. Huang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
import types

sys.path.insert(0, os.path.abspath('../../src'))

# Prevent the metrics package __init__ from eagerly importing every submodule
# during autodoc, which creates a circular import when scTimeBench.database is
# imported for documentation.
_metrics_pkg_path = os.path.abspath('../../src/scTimeBench/metrics')
_metrics_stub = types.ModuleType('scTimeBench.metrics')
_metrics_stub.__path__ = [_metrics_pkg_path]
_metrics_stub.__package__ = 'scTimeBench.metrics'
_metrics_stub.__file__ = os.path.join(_metrics_pkg_path, '__init__.py')
sys.modules.setdefault('scTimeBench.metrics', _metrics_stub)

extensions = ['sphinx.ext.autodoc']
autodoc_default_options = {
	'members': True,
	'undoc-members': True,
	'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_theme_options = {
	'logo_only': False,
}
html_css_files = ['custom.css']


def autodoc_process_docstring(app, what, name, obj, options, lines):
	if name == 'scTimeBench.metrics.base.BaseMetric.eval':
		lines[:] = [
			'Evaluation function that handles the calling of submetrics if applicable.',
			'',
			'Basically it happens as follows:',
			'',
			'1. If there are submetrics defined, we create an instance of each submetric.',
			'2. We call the _eval function of each submetric.',
			'3. From this _eval function, we further call the _submetric_eval function that each subclass must implement.',
		]
	elif name == 'scTimeBench.trajectory_infer.base.BaseTrajectoryInferMethod.infer_trajectory':
		lines[:] = [
			'Infer the trajectory using the kNN graph-based method.',
			'',
			'1. Separate each embedding by time.',
			'2. Find the k nearest neighbors in the next time point embedding space.',
			'3. Consolidate the cell types per time point based on the kNN results.',
		]


def setup(app):
	app.connect('autodoc-process-docstring', autodoc_process_docstring)
