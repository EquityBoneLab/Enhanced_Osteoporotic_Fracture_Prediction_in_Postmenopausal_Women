import numpy as np
from functools import partial
from matplotlib import pyplot
import seaborn as sns

import github_XGB_GRS_bayes_plot,github_XGB_GRS_grid_plot,githubXGB_noGRS_bayes_plot,github_XGB_noGRS_grid
if __name__ == '__main__':
    models = [
        {
            'label': 'XGBoost (FRAX) with grid search method',
            'model': partial(github_XGB_noGRS_grid.get_data, plot=False)
        },
        {
            'label': 'XGBoost (FRAX) with Bayesian tuning method',
            'model': partial(github_XGB_noGRS_bayes_plot.get_data, plot=False)
        },
        {
            'label': 'XGBoost (FRAX + GRS) with grid search method',
            'model': partial(github_XGB_GRS_grid_plot.get_data, plot=False)
        },
        {
            'label': 'XGBoost (FRAX + GRS) with Bayesian tuning',
            'model': partial(github_XGB_GRS_bayes_plot.get_data, plot=False)
        }
    ]

    labels = ['Model 1',
              'Model 2',
              'Model 3',
              'Model 4']

    sns.set(style="ticks")
    pyplot.figure(figsize=(18, 18))
    sns.despine()
    types = ('g', 'darkorange', 'b', 'm')
    for i, m in enumerate(models):
        fpr, tpr, thresh, auc = m['model']()
        pyplot.plot(fpr, tpr, types[i], lw =4, label="{}, AUC={:.3f}".format(labels[i], auc))
    pyplot.plot([0, 1], [0, 1], color='gray', lw=1.0, linestyle='--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.0])
    pyplot.xticks(np.arange(0.0, 1.1, step=0.1),fontsize=47)
    pyplot.xlabel('1-Specificity (False Positive Rate)', fontsize=48, fontname='Arial')
    pyplot.yticks(np.arange(0.0, 1.1, step=0.1),fontsize=47)
    pyplot.ylabel('Sensitivity (True Positive Rate)', fontsize=48, fontname='Arial')
    pyplot.legend(loc='lower right', fontsize=50)

