# CT421-A1
Submission for CT421 - Artificial Intelligence Assignment 1, 2026

main.py is the main GA implementation which uses functions from utils.py
run_stats.py is the same implementation with summary statistics, you can specify how many runs to complete
param_cv.py attempts to use k-fold cross-validation to select optimal GA parameters from a specified search grid

TO RUN:
    -   main.py: python src/main.py <data_file> [seed]
    -   run_stats.py: python tests/run_stats.py <data_file> [#runs]
    -   param_cv.py: python tests/param_cv.py <data_file> [#runs] [#folds]
