# %% [markdown]
# # Basic seep Analysis
#
# This notebook returns basic statistics and visualizations for any kind of sweep saved as a wandb run group or as a local Hydra multirun.
# It is designed to be flexible and work with any dataset, model, and training regime, as long as the data is structured in a compatible way.
#
# **Data Sources:**
# - **wandb**: Fetches run data from Weights & Biases API
# - **local**: Loads data from local outputs/ directory (Hydra multirun)
#
# **Visualizations:**
# 1. Histogram of accuracy, with possible inclusion of robust and MIA accuracy if available.
# 2. Mean accuries with std error bars. 
# 3. Scatter plot of (clean, rob, MIA) accuracy against loss (as used for training) on validation set.
# 4. Best run according to validation loss, and its corresponding accuracies.
# 5. Final table showing best, mean, and std of accuracies across all runs of the group.
#
# **Lightweight and Flexible:**
# - Focuses on core statistics (result oriented) and basic vizualisations
# - Allows for correction of std in case of smaller actual number of independent runs (e.g. dead hyperparameter like tracking.random_state that do not change the runs)
# - Imports from from visualize_distributions.py and mia_analysis.py for the needed visualizations and accuracy calculations.