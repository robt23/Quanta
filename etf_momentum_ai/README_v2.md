# ETF Momentum AI - Version 2

This is Version 2 of the ETF Momentum AI pipeline. This version introduces updated data preparation, feature engineering, and model training scripts for improved crash prediction and momentum-based ETF strategies.

## Pipeline Overview & Script Order

Run the following scripts in order:

1. **data_prep_v2.py**
   - Downloads SPY and VIX data, computes returns, volatility, momentum, and technical indicators. Generates the main dataset and crash labels for model training.

2. **feature_engineering_v2.py**
   - Performs advanced feature engineering on the dataset, creating additional features and feature interactions to improve model performance.

3. **regime_feature_v2.py**
   - Adds market regime features (e.g., bull/bear/neutral regime indicators) to the dataset using PCA and KMeans clustering for regime-aware modeling.

4. **hyperopt_lgbm.py**
   - Runs hyperparameter optimization for the LightGBM model using Optuna, searching for the best parameters with a time-series cross-validation and F2-score maximization.

5. **train_final_lgbm.py**
   - Trains the final LightGBM model using the engineered features and optimized parameters. Applies probability calibration and saves the trained model for inference.

6. **threshold_scan.py**
   - Scans different probability thresholds for the crash prediction model to optimize signal generation and risk management. Saves the best threshold for use in backtesting.

7. **evaluate_model.py**
   - Evaluates the trained model's performance on validation/test data, reporting metrics such as confusion matrix, PR AUC, ROC AUC, F2 score, precision, and recall.

8. **core_alpha_beta_core2_scan.py**
   - Performs a grid search over core/alpha/beta/core2 parameters for the backtest strategy, saving the best configuration and results.

9. **momentum_backtest.py**
   - Runs a backtest of the momentum-based ETF strategy using the model's crash predictions, engineered features, and optimal parameters. Produces performance plots and metrics.

## Notes
- All scripts are located in the `etf_momentum_ai/` directory.
- Intermediate and final datasets are saved in the `datasets/` subfolder.
- Model artifacts and results are saved in the `models/` and `results/` subfolders, respectively.
- Additional analysis scripts and utilities can be found in the `analysis_scripts/` and `version1/` subfolders.

For more details, see comments in each script.
