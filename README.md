# nonlinear-weight-updates

Provides data and implementations (in progress) related to the paper "On the benefits of non-linear weight updates"

## Data

Folder deepobs_results_medium_budget includes the search and training data for the key results presented in the paper. This is in the format used in https://github.com/fsschneider/DeepOBS .

## Implementations

Folder includes an implementation that works with Keras. This version selects the weights to apply the NL function to based purely on the variable name. (In fact, it looks for variables with 'ernel' in the name due to worries about capitalisation.) A more rigourous implementation would allow a subset of variables to be identified at initialisation time.

(Note that this is not the implementation used for the paper, since DeepObs is written for Tensorflow 1.15.)
