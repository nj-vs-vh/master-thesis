"""
new code should use submodules directly:
>>> import modules.plots.specific as specific_plots
>>> specific_plots.plot_something_specific()

imports are here to keep methodoogical notebooks working as they use 'import modules.plots'
"""

from .deconvolution import (
    plot_convolution,
    plot_data_similarity_test,
    plot_mean_n_estimation,
    plot_mean_n_estimation_assessment,
    plot_S_j_marginal_normality_assessment,
    plot_S_j_pairwise_normality_assessment,
    plot_bayesian_mean_estimation,
)


__all__ = [
    'plot_convolution',
    'plot_data_similarity_test',
    'plot_mean_n_estimation',
    'plot_mean_n_estimation_assessment',
    'plot_S_j_marginal_normality_assessment',
    'plot_S_j_pairwise_normality_assessment',
    'plot_bayesian_mean_estimation',
]
