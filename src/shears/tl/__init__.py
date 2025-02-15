import itertools
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from tqdm import tqdm

from shears._util import fdr_correction, process_map


def _test_cell(formula, df, family="binomial"): #!! TODO: also assign "gaussian" now it always calls "binomial" ... !!!
    mod = smf.glm(formula, data=df, family=sm.families.Binomial() if family == "binomial" else sm.families.Gaussian())
    res = mod.fit()
    return res.pvalues["cell_weight"], res.params["cell_weight"]

def _test_cell_cox(formula, df):
    warnings.filterwarnings("ignore")
    cph = CoxPHFitter()
    cph.fit(df, duration_col="OS_Time", event_col="OS_Status", formula=formula)
    return cph.summary.loc["cell_weight", "p"], cph.summary.loc["cell_weight", "coef"]


def shears(
    adata_sc,
    adata_bulk,
    *,
    dep_var="",
    covariate_str="",
    inplace=True,
    cell_weights_key="cell_weights",
    key_added="shears",
    family="binomial",
    duration_col="OS_time",
    event_col="OS_status",
    n_jobs=None,
):
    """
    Compute score for each cell.

    Stores the scores in adata_sc.obs[key_added] unless inplace is False.

    Parameters
    ----------
    covariate_str
        covariates to add, e.g. `+ age + C(gender)`
    family
        model family: 'binomial', 'gaussian', or 'cox'.
    """
    if family == "cox":
        formula = f"cell_weight {covariate_str}"
        keep_cols = [c for c in adata_bulk.obs.columns if c in duration_col + event_col + covariate_str]
    else:
        formula = f"{dep_var} ~ cell_weight {covariate_str}"
        keep_cols = [c for c in adata_bulk.obs.columns if c in dep_var + covariate_str]
        
    assert "cell_weight" not in keep_cols, "cell_weight is a reserved column name"
    bulk_obs = adata_bulk.obs.loc[:, keep_cols].copy()
    # TODO once I get family="gaussian" to work no need to rename columns but use provided colnames in fun!
    bulk_obs = bulk_obs.rename(columns={duration_col: "OS_Time", event_col: "OS_Status"})

    print(formula)

    def _df_iter():
        for c in adata_sc.obs_names:
            tmp_df = bulk_obs.copy()
            tmp_df["cell_weight"] = adata_sc.obsm[cell_weights_key].loc[c, adata_bulk.obs_names]
            yield tmp_df

    if family == "cox":
        test_func = _test_cell_cox
    else:
        test_func = _test_cell

    res = process_map(
        test_func,
        itertools.repeat(formula),
        _df_iter(),
        total=adata_sc.shape[0],
        chunksize=100,
        max_workers=n_jobs,
        tqdm_class=tqdm,
    )

    return pd.DataFrame(res, index=adata_sc.obs_names, columns=["pvalue", "coef"])


def get_scaling_factor(adata_sc, cell_type_col, *, weight_col="n_genes", callback=np.median):
    """Get scaling factors per cell-type.

    Corrects for the number of cells per dataset and the mRNA content of cells.

    Parameters
    ----------
    adata_sc
        single cell AnnData
    cell_type_col
        column in adata_sc.obs that holds cell-type information
    weight_col
        column in adata.obs that holds a cell weight to correc for mRNA bias. Viable options are
        the number of detected genese or the number of total counts per cell. See Dietrich et al. for
        more details
    callback
        Aggregate the values in weight_col by cell_type using this function.
    """
    # multiply weights with the number of cells (if there are many cells of a type in the single-cell dataset,
    # the factors will distribute across more cells and the weights dilute)
    n_cells = adata_sc.obs[cell_type_col].value_counts()

    # divide by a scaling factor to account for mRNA contents. Gene expression is normalized,
    # therefore e.g. Neutrophils (low mRNA) and Macrophages (high mRNA) have the same amount of normalized counts.
    # However, if they get the same weight, there will be more Neutrophils in the bulk sample,
    # because they contribute less mRNA to the bulk sample.
    cell_weight = adata_sc.obs[weight_col].groupby(adata_sc.obs[cell_type_col]).agg(callback)

    factor = n_cells / cell_weight
    return factor.reindex(adata_sc.obs[cell_type_col])


def cell_type_fractions(adata_sc, cell_type_col, as_fraction=True, bias_correction=True):
    """Compute cell-type fractions from cell weights"""
    cell_weights = adata_sc.obsm["cell_weights"]

    if bias_correction:
        cell_weights = cell_weights.multiply(get_scaling_factor(adata_sc, cell_type_col).values, axis=0)

    df = cell_weights.groupby(adata_sc.obs[cell_type_col]).agg(sum)

    if as_fraction:
        df = df.div(df.sum(axis=0).to_dict())

    return df


def shears_stats(adata_sc, res, groupby, batch_key: str):
    """Compute statistics on shears results"""
    df = res.assign(weight=lambda x: x["coef"] * -np.log10(x["pvalue"])).assign(
        group=adata_sc.obs[groupby], replicate=adata_sc.obs[batch_key]
    )
    df = (
        df.groupby(["group", "replicate"], observed=True)
        .agg(mean_weight=pd.NamedAgg(column="weight", aggfunc=np.mean))
        .reset_index()
    )
    stats_df = (
        df.groupby("group")
        .apply(
            lambda x: pd.Series(
                {
                    "median_weight": np.median(x["mean_weight"]),
                    # **scipy.stats.ttest_1samp(x["mean_weight"], 0)._asdict(),
                    **scipy.stats.wilcoxon(x["mean_weight"])._asdict(),
                }
            )
        )
        .reset_index()
        .pipe(fdr_correction)
        .sort_values("pvalue")
    )
    return df.merge(stats_df, on="group", how="left")
