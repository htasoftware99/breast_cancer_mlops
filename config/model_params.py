from scipy.stats import uniform

LOGISTIC_REGRESSION_PARAMS = {
    'C'      : [0.01, 0.1, 1, 10, 100],
    'solver' : ['lbfgs', 'liblinear'],
    'max_iter': [100, 500, 1000]
}

GRID_SEARCH_PARAMS = {
    'cv'      : 5,
    'n_jobs'  : -1,
    'verbose' : 2,
    'scoring' : 'roc_auc'  
}