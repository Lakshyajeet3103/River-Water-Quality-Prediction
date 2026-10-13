# Expanded scikit-learn BOD demonstration

Rows: {'train': 38, 'validation': 5, 'test': 17}; excluded by boundary/gap: 4.

Test MAE: 0.5360 mg/L; RMSE: 0.7716 mg/L; R²: 0.3558.

Training-mean test MAE: 0.9562 mg/L.

The friend's original model/settings are unchanged: 11 inputs, 8 ReLU hidden neurons, L-BFGS, alpha=1, seed=42. Scaling is fitted only on training rows. No hyperparameters were tuned on the test set.

The saved model was reloaded and reproduced the saved test predictions. This small retrospective demonstration does not establish water safety or future/unseen-region performance. Earlier and expanded metrics use different test sets.
