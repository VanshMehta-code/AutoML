# 10 Glossary

**ACO (ant colony optimisation).** A search method in which simulated "ants" build candidate solutions, and successful choices leave pheromone that guides later ants. Used here to choose a feature subset.

**AUC (ROC AUC).** Area under the receiver operating characteristic curve. The probability that a random positive example gets a higher score than a random negative one. Threshold-free, so it measures ranking quality.

**Artifact.** The saved `model.joblib` plus its `report.json` and environment record. The unit that gets deployed.

**Candidate.** One (model family, hyperparameters) pair under evaluation.

**CV (cross-validation).** Splitting training data into folds, training on all but one and scoring on the held-out one, then repeating. Gives an estimate of performance without touching the test set.

**Elite.** The best candidates kept as parents for the next generation.

**Encoding.** Turning a non-numeric column into numbers. Here: one-hot for few levels, target encoding for many.

**Ensemble.** A weighted combination of several models' predicted probabilities.

**Evaluator.** The component that scores a candidate with cross-validation, caches the result, and supports subsampling and feature masks.

**Evolutionary search.** Search that keeps a population of candidates, selects the better ones, and produces new candidates by crossover and mutation.

**Fidelity.** How much of the data (or how expensive a computation) a score is based on. Low-fidelity scores are cheap and noisier.

**Feature graph.** A graph whose nodes are features and whose weights capture relevance to the target and redundancy with other features.

**FittedModel.** The object that holds everything inference needs: preprocessor, mask, members, weights, and input schema.

**Holdout / test set.** Rows set aside before any learning and used once, for the final honest score.

**Identifier column.** A column with almost one distinct value per row (IDs, free text). Dropped, since it cannot generalise.

**Leakage.** Information from outside the training data, typically the test set or the future, influencing training. Makes scores look better than they will be in production.

**Metadata.** Statistics describing the dataset and each column, used to make cleaning and search decisions.

**Mutual information (MI).** A measure of how much knowing one variable reduces uncertainty about another. Captures non-linear relationships. Used as feature relevance.

**Multiclass.** A classification task with three or more classes. Scored by one-vs-rest AUC.

**OOF (out-of-fold).** Predictions for each training row made by a model that did not train on that row. Used for ensemble weights without leakage.

**Pheromone.** In ACO, a number per option that records how good it has been, and biases future choices.

**Preprocessor.** The fit/transform estimator that cleans and encodes features. Learned on training data only.

**Redundancy.** Two features carrying nearly the same information. Measured here by |Spearman| correlation.

**Refit.** Training a chosen model again on more data, with the same settings.

**Spearman correlation.** Correlation computed on ranks. Detects any monotonic relationship, not just linear ones.

**Successive halving.** Evaluate many candidates cheaply, keep the best fraction, and evaluate only those more thoroughly.

**Target encoding.** Replacing each category with the average target value for that category, cross-fitted to limit overfitting.

**Winsorisation (clipping).** Limiting extreme values to a chosen range, here based on the IQR, instead of removing rows.
