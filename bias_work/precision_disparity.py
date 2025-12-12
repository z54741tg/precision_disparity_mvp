import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt


class PrecisionDisparityAnalyser:
    """
    Compute precision by subgroup and derive:
      - relative disparity ratios  (precision / reference precision)
      - absolute percentage-point gaps (precision - reference precision)

    Intended for linkage evaluation where each row is a *made link*
    classified as True Positive (TP) or False Positive (FP).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        truth_col: str = "link_truth",
        positive_label: str = "TP",
        negative_label: str = "FP",
        group_vars: Optional[List[str]] = None,
        min_linked: int = 30,
        dropna_groups: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        df : DataFrame
            Input dataframe. Each row should represent a *candidate link*
            that has been classified as TP or FP.
        truth_col : str
            Column containing TP / FP labels.
        positive_label : str
            Value in `truth_col` representing a true positive.
        negative_label : str
            Value in `truth_col` representing a false positive.
        group_vars : list of str, optional
            Columns to use as subgroup variables.
        min_linked : int
            Minimum number of links in a group (TP+FP) to flag as unstable.
        dropna_groups : bool
            If True, drop rows where any group_var is missing.
        """
        self.df = df.copy()
        self.truth_col = truth_col
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.group_vars = group_vars or []
        self.min_linked = min_linked
        self.dropna_groups = dropna_groups

        self._dropped_non_binary: Optional[int] = None
        self._dropped_missing_groups: Optional[int] = None

    # ------------- internal helpers -----------------

    def _validate_inputs(self) -> None:
        if self.truth_col not in self.df.columns:
            raise ValueError(f"truth_col '{self.truth_col}' not in dataframe")

        missing_groups = [g for g in self.group_vars if g not in self.df.columns]
        if missing_groups:
            raise ValueError(f"Missing group columns: {missing_groups}")

        # Keep only TP / FP rows
        mask = self.df[self.truth_col].isin([self.positive_label, self.negative_label])
        self._dropped_non_binary = int(len(self.df) - mask.sum())
        self.df = self.df[mask].copy()

        # Optionally drop missing subgroup values
        if self.dropna_groups and self.group_vars:
            before = len(self.df)
            self.df = self.df.dropna(subset=self.group_vars)
            self._dropped_missing_groups = int(before - len(self.df))
        else:
            self._dropped_missing_groups = 0

        if self.df.empty:
            raise ValueError("No rows left after filtering to TP/FP and dropping NA groups.")

    def _add_indicators(self) -> None:
        self.df["is_tp"] = (self.df[self.truth_col] == self.positive_label).astype(int)
        self.df["is_fp"] = (self.df[self.truth_col] == self.negative_label).astype(int)

    @staticmethod
    def _precision_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        grouped = df.groupby(group_col).agg(
            tp=("is_tp", "sum"),
            fp=("is_fp", "sum"),
        ).reset_index()

        grouped["linked"] = grouped["tp"] + grouped["fp"]
        grouped["precision"] = grouped["tp"] / grouped["linked"].replace(0, np.nan)
        return grouped

    @staticmethod
    def _choose_reference(
        rates_df: pd.DataFrame,
        group_col: str,
        mode: str = "max_precision",
        explicit_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pick the reference group for disparity calculations.

        mode="max_precision"  -> group with highest precision
        mode="explicit"       -> use user-specified group name
        """
        df = rates_df.copy()

        if mode == "explicit":
            if explicit_group is None:
                raise ValueError("reference_mode='explicit' requires `explicit_group`.")
            if explicit_group not in df[group_col].astype(str).tolist():
                raise ValueError(f"explicit_group '{explicit_group}' not found in {group_col}")
            ref_row = df[df[group_col] == explicit_group].iloc[0]
        elif mode == "max_precision":
            ref_row = df.loc[df["precision"].idxmax()]
        else:
            raise ValueError(f"Unknown reference_mode '{mode}'")

        return {
            "group": ref_row[group_col],
            "precision": float(ref_row["precision"]),
        }

    @staticmethod
    def _add_disparities(
        rates_df: pd.DataFrame,
        group_col: str,
        ref_info: Dict[str, Any],
        min_linked: int,
    ) -> pd.DataFrame:
        out = rates_df.copy()

        ref_group = ref_info["group"]
        ref_precision = ref_info["precision"]

        out["reference_group"] = ref_group
        out["reference_precision"] = ref_precision

        out["precision_dr"] = out["precision"] / ref_precision
        out["precision_ppg"] = out["precision"] - ref_precision

        out["low_volume_flag"] = out["linked"] < min_linked
        out["is_reference"] = out[group_col] == ref_group

        return out

    # ------------- public API -----------------

    def run(
        self,
        reference_mode: str = "max_precision",
        reference_groups: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        if not self.group_vars:
            raise ValueError("No group_vars provided.")

        self._validate_inputs()
        self._add_indicators()

        results = []

        for var in self.group_vars:
            rates = self._precision_by_group(self.df, var)

            if reference_mode == "explicit":
                explicit_group = None if reference_groups is None else reference_groups.get(var)
                ref_info = self._choose_reference(
                    rates_df=rates,
                    group_col=var,
                    mode="explicit",
                    explicit_group=explicit_group,
                )
            else:
                ref_info = self._choose_reference(
                    rates_df=rates,
                    group_col=var,
                    mode="max_precision",
                )

            enriched = self._add_disparities(
                rates_df=rates,
                group_col=var,
                ref_info=ref_info,
                min_linked=self.min_linked,
            )

            enriched["variable"] = var
            enriched = enriched.rename(columns={var: "group"})
            results.append(enriched)

        out = pd.concat(results, ignore_index=True)

        col_order = [
            "variable",
            "group",
            "tp",
            "fp",
            "linked",
            "precision",
            "reference_group",
            "reference_precision",
            "precision_dr",
            "precision_ppg",
            "low_volume_flag",
            "is_reference",
        ]
        col_order = [c for c in col_order if c in out.columns] + [
            c for c in out.columns if c not in col_order
        ]
        return out[col_order]

    def summary(self) -> str:
        lines = []
        lines.append("Precision disparity analysis")
        lines.append("-----------------------------------")
        lines.append(
            f"Rows with non-TP/FP labels dropped: "
            f"{self._dropped_non_binary if self._dropped_non_binary is not None else 'not run'}"
        )
        lines.append(
            f"Rows dropped due to missing group values: "
            f"{self._dropped_missing_groups if self._dropped_missing_groups is not None else 'not run'}"
        )
        lines.append(f"Minimum linked volume for stability flag: {self.min_linked}")
        if self.group_vars:
            lines.append(f"Grouping variables analysed: {', '.join(self.group_vars)}")
        return "\n".join(lines)


def plot_precision_disparity(
    results: pd.DataFrame,
    variable: str,
    metric: str = "precision_dr",
    include_reference: bool = True,
) -> None:
    """
    Simple bar plot for one variable from the results table.
    """
    df_var = results[results["variable"] == variable].copy()

    if not include_reference:
        df_var = df_var[~df_var["is_reference"]]

    df_var = df_var.sort_values(metric)

    plt.figure(figsize=(8, 4))
    plt.bar(df_var["group"].astype(str), df_var[metric])
    plt.axhline(1.0 if metric == "precision_dr" else 0.0, linestyle="--")
    plt.title(f"{variable}: {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()