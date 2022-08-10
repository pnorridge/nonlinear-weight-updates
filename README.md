# nonlinear-weight-updates

Provides data and implementations (in progress) related to the paper *On the Benefits of Non-Linear Weight Updates* http://arxiv.org/abs/2207.12505

Key result in a sentence: You can improve generalisation by applying a non-linear function to gradients before passing them to the optimiser. (A well-chosen function, of course.)

The motivation: If you have correlated inputs to a node, the best weight set is one that balances the contribution from these inputs. This weight set is more robust to noise. So, we pick a function that encourages more balance.

The geometric picture: Correlated inputs are intimately related to flat minima.  But, good performance is not just about finding flat minima, it's also about making sure the model ends up in a good location on this plane.

## Data

Folder deepobs_results_medium_budget includes the search and training data for the key results presented in the paper. This is in the format used in https://github.com/fsschneider/DeepOBS.

Note that results sets labelled '_nodecay' are those explicitly implementing weight decay rather than L2 regularisation. The L2 factor in the test case is set to zero and replaced with weight decay in the optimizer (see paper for discussion). (I know, the naming is poor - and will be fixed...)

## Implementations

Folder includes an implementation that works with Keras. This version selects the weights to apply the NL function to based purely on the variable name. (In fact, it looks for variables with 'ernel' in the name due to worries about capitalisation.) A more rigourous implementation would allow a subset of variables to be identified at initialisation time.

(Note that this is not the implementation used for the paper, since DeepObs is written for Tensorflow 1.15.)
