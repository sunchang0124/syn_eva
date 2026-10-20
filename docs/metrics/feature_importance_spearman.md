# `feature_importance_spearman` (utility / extended)

**Scope:** table-level. **Requires real data:** True. **Data types:** ['static'].

## What it measures

Whether a model trained on synthetic data relies on the same features as one
trained on real data. For each utility task, a random forest is fitted to
predict the target on each dataset, and the two feature-importance rankings are
compared. A synthetic dataset can reach a good TSTR score while getting there
through different features; this metric catches that.

## How it's computed

For each task, the features are the task's `features` override, or every
column except the target. Duplicates and the target are removed, and only
numeric, categorical and boolean features are kept, since those are the ones
the model can use; ID and datetime columns are left out, and a note names any
override feature dropped this way. Missing values (`NaN`, `None`
and `pd.NA` alike) are imputed: numeric features with their mean, then
standardized; categorical and boolean features with their most frequent value,
then one-hot encoded. A feature that is entirely missing becomes a constant
column. A random forest (100 trees, fixed seed; classifier or
regressor depending on the task type) is fitted separately on the real and the
synthetic data.

Each forest gives a mean decrease in impurity (MDI) per encoded column. Real
and synthetic data can have different category sets, so their one-hot columns
don't line up. The importances are therefore summed back to the original
features first. For a feature $X_j$ encoded into one-hot columns $C_j$,

$$
\text{Imp}(X_j) = \sum_{c \in C_j} \text{MDI}(c) .
$$

A split on a one-hot column is a split on $X_j$, so this sum is the total
impurity decrease from splits on $X_j$, which is the MDI definition applied to
the original variable. Numeric features map to a single column.

A feature that is constant or entirely missing on one side is constant there
after imputation, so the forest never splits on it and it scores 0; a note
names the feature and the side. If that is true on both sides, the feature is
left out of the ranking with a note: its importance is 0 on both sides by
construction, and that tied pair would pull $\rho$ towards 1 without saying
anything about the data.

With importance vectors $a$ (real) and $b$ (synthetic) over the same $p$
features, the task's value is Spearman's rank correlation
$\rho = \operatorname{corr}(\operatorname{rank}(a), \operatorname{rank}(b))$.
The table-level value is the mean $\bar\rho$ over tasks, and

$$
\text{score} = \frac{\bar\rho + 1}{2} .
$$

A task is skipped, with a note, when its target is missing from the real data
or when fewer than 2 features are left after the steps above. If a side's importances are all
equal (for example, a constant target leaves every importance at 0), $\rho$ is
undefined; it is set to 0 and a note is added. The metric is skipped when no
task can be evaluated.

## When to use it

Use it next to `tstr_suite` when you need the synthetic data to support the
same explanations as the real data, not just the same accuracy. Keep in mind:

- **Few features give a coarse signal.** With $p$ features, $\rho$ takes only a handful of values; with 2 features it is either $-1$ or $1$. Treat single-task results on small feature sets with caution.
- **MDI favours some features.** Impurity importance is biased towards numeric and high-cardinality categorical features, which offer more split points (Strobl et al., 2007). The bias does not cancel between the two sides when they differ in a feature's number of distinct values: a categorical feature with fewer levels in the synthetic data has fewer split points there, so its synthetic importance can drop for that reason alone. A lower $\rho$ can therefore partly reflect lost categories rather than a different relationship with the target. Neither ranking should be read as a ground-truth explanation.
- **Importances are in-sample.** Both forests are scored on the data they were trained on, so features that only help the model fit noise can still rank above zero.
- **Missing categories matter.** If a category that drives the target is absent from the synthetic data, the feature's importance drops and $\rho$ goes down. This is intended: the synthetic data no longer supports the same model.

## Output scalars

- `score`: $(\bar\rho + 1) / 2$, from 0 to 1, higher is better.
- `spearman_rho`: $\bar\rho$, the mean rank correlation over tasks, from $-1$ to 1; 1 means identical importance orderings.

`per_column` has one entry per evaluated task, keyed by target, with
`spearman_rho` and `n_features` (the number of features that were ranked).

## Citation

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Louppe, G., Wehenkel, L., Sutera, A., & Geurts, P. (2013). Understanding
variable importances in forests of randomized trees. *Advances in Neural
Information Processing Systems 26*, 431–439.

Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in
random forest variable importance measures: Illustrations, sources and a
solution. *BMC Bioinformatics*, 8, 25.

Spearman, C. (1904). The proof and measurement of association between two
things. *The American Journal of Psychology*, 15(1), 72–101.
