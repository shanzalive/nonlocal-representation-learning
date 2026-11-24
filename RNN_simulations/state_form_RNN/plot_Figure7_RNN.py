#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 7c-h and Fig. S6: plots the co-update analysis of the performance-matched networks.
#
# Loads the two results files written by coupdate_analysis_RNN.py (empirical and dense sampling)
# and draws Figure 7 (main: context-dependent co-updates and their link to generalization,
# empirical and dense sampling) and Fig. S6 (co-updates for the irrelevant context feature, both sampling schemes).
# Prints all statistics shown in the figures, and the house-match statistic reported in the text.
# Run from inside RNN_simulations/state_form_RNN/ (the network fits are found via a relative path).
# Input:  <save_dir>/acc_curve_mse_empirical.npy and acc_curve_mse_dense.npy
# Output: Figure7_main_acc_curve_mse.pdf, Figure7_supp_irrelevant_acc_curve_mse.pdf
#
# Author: Shany Grossman

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_rel
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import os
import json
import matplotlib.transforms as transforms
import warnings
import gc

# Folder with the participant-level behavioural summary files, found by searching upwards
# from this script (works on the cluster, on a mounted drive and in the code repository).
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour folder not found above ' + os.path.abspath(__file__))
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)   # mini blocks wihtout any probes                                                                                          
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide', category=RuntimeWarning)  # mini blocks wihtout the relevant vaue levels     

simulations = ['acc_curve_mse']
# networks fitted to participants (fits are needed only to know which participants have a fit)
data_dir = '../state_form_RNN_outputs/NN_activations/gradientDescentPerfromanceMatched/grid_search_fits/acc_curve_mse_w15_fc_gain_rnn_gain_rl_temp_1_greedy_choice_10_seeds_9_grid_points/'

# Same underlying storage, mounted at a different path on the Mac vs the cluster.
_MAC_ROOT     = '/Users/grossman/Volumes/STATE_FORMATION'
_CLUSTER_ROOT = '/home/mpib/grossman/STATE_FORMATION'
_ROOT = _MAC_ROOT if os.path.isdir(_MAC_ROOT) else _CLUSTER_ROOT
output_basedir = _ROOT + '/RNN_simulations/state_form_RNN_outputs/coupdateAnalysis/'

# Must match the settings used when running coupdate_analysis_RNN.py (they set the file names).
k = 1                                  # step size of dense sampling
alignmentReferenceSet = 'probe_set'    # 'probe_set' or 'all_states' (results are qualititavely unchanged by either)
doProcrustesAlignment = True
_align_suffix = '' if alignmentReferenceSet == 'probe_set' else '_alignAll'
_proc_suffix = '' if doProcrustesAlignment else '_noProc'

# Participants: full + partial learners from learners_dict.json
with open(SUMMARY_DIR + 'learners_dict.json') as _f:
    _learners_dict = json.load(_f)
included_subjects = [s.replace('-', '') for s in sorted(_learners_dict['full_and_partial'])]

for simulation in simulations: # for each particpant-matched network
    sim_included_subjects = [s for s in included_subjects
                              if os.path.isfile(os.path.join(data_dir, s + '_fitted_params.npy'))]
    print(f'Including {len(sim_included_subjects)} subjects with a network fit.')
    save_dir = output_basedir + 'gradientDescentPerfromanceMatched/' + simulation + '/heldOutSeed'

    # Load data summary files for both sampling schemes ( _empirical and _dense):
    results_emp   = np.load(f"{save_dir}/{simulation}{_align_suffix}{_proc_suffix}_empirical.npy", allow_pickle=True).item()
    _dense_tag = f'_denseStep{k}' if k != 1 else ''
    results_dense = np.load(f"{save_dir}/{simulation}{_align_suffix}{_proc_suffix}_dense{_dense_tag}.npy", allow_pickle=True).item()

    # Generalization classes (generalized / not generalized / saturated) and the per-participant
    # mean co-update of each class come from coupdate_analysis_RNN.py (same rule for both sampling
    # schemes). Each paired comparison below uses the participants with segments in both classes.

    # Compute and print visual-/input-dep co-update (i.e. house match) : co-update of the cross-context pairs whose two states share the same house type vs pairs with different house types,
    # mean over learning segments per participant, paired t-test across participants.

    for _layer in ['RNN_hidden1', 'hidden2']:
        _same = np.nanmean(results_emp[_layer]['cross_ctx_same_house_coup_all'], axis=1)
        _diff = np.nanmean(results_emp[_layer]['cross_ctx_diff_house_coup_all'], axis=1)
        _ok = np.isfinite(_same) & np.isfinite(_diff)
        _t, _p = ttest_rel(_same[_ok], _diff[_ok])
        _d = np.mean(_same[_ok] - _diff[_ok]) / np.std(_same[_ok] - _diff[_ok], ddof=1)
        print(f'[house match] Empirical sampling | {_layer} | same vs different house: '
              f't({_ok.sum() - 1})={_t:.2f}, p={_p:.2g}, d={_d:.2f}')

    subjects = sim_included_subjects

    # Plot settings
    plt.rcParams.update({
        'figure.dpi': 150,   # screen DPI 
        'savefig.dpi': 600,
        'font.family': 'Arial',
        'pdf.fonttype': 42,
        'font.size': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 1,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.major.size': 3,
        'ytick.major.size': 3})


    #%% Plot Figure7 helper functions

    # Helper functions for significance annotations
    def significance_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    def format_p_value(p):
        if p < 0.001:
            return "p<0.001"
        else:
            return f"p={p:.3f}"

    #%% Figure 7 (empirical and dense sampling) and Fig. S6 (irrelevant context feature)

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.ticker import FuncFormatter
    from scipy.stats import ttest_rel, pearsonr, spearmanr
    import itertools
    from scipy.stats import sem
    from matplotlib.lines import Line2D

    def _no_leading_zero(v, _):
        s = f'{v:.1f}'
        if s.startswith('0.'):   return s[1:]
        if s.startswith('-0.'): return '-' + s[2:]
        return s
    _fmt_nolz = FuncFormatter(_no_leading_zero)

    # Settings shared by all panels
    layers_ts        = ['RNN_hidden1', 'hidden2']   # timeseries panels: both layers
    layer_for_panels = 'RNN_hidden1'                # paired swarm panels: RNN layer only

    base_same = "#5BC8C0"
    base_diff = "darkblue"
    base_dep  = "k"
    base_mag  = '#8B6FB5'

    context_panel_type = ("Context feature",
                           "same_context_coup_all", "diff_context_coup_all", "context_dep_coup_all")
    irrel_panel_type    = ("Irrelevant context feature",
                            "same_irrel_context_coup_all", "diff_irrel_context_coup_all",
                            "context_irrel_dep_coup_all")

    doPlotMeanDifference = True
    connect_subjects = True

    conditions_3 = ["no_generalization", "generalization", "saturated_learning"]
    conditions_2 = ["no_generalization", "generalization"]

    cond_labels = {
        "generalization": "Generalization",
        "no_generalization": "No\ngeneralization",
        "saturated_learning": "Saturated\nlearning"}

    panel_colors = {
        'corr_diff_context': 'darkblue',
        'corr_same_context': '#5BC8C0',
        'delta_corr': 'k'}

    columns = [("Empirical sampling",     results_emp,   True),
               ("Dense sampling",  results_dense, False)]

    def _add_ts_legend(fig, y_anchor):
        # A single legend for the entire figure 
        _legend_handles = [
            Line2D([0], [0], color=base_same, lw=2, ls='-',  label='Same context, RNN layer'),
            Line2D([0], [0], color=base_diff, lw=2, ls='-',  label='Cross-context, RNN layer'),
            Line2D([0], [0], color=base_same, lw=2, ls=':',  label='Same context, FC layer'),
            Line2D([0], [0], color=base_diff, lw=2, ls=':',  label='Cross-context, FC layer')]
        fig.legend(handles=_legend_handles, fontsize=7, frameon=False, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, y_anchor))

    DENSE_SMOOTH_WIN = 10  # running-average window (in trials) applied to dense-sampling timeseries only

    def _smooth_running(arr, w):
        """Centered moving average along axis=1 (trials/segments), nan-safe."""
        if w <= 1:
            return arr
        arr = np.asarray(arr, dtype=float)
        out = np.full_like(arr, np.nan)
        half = w // 2
        n = arr.shape[1]
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            out[:, i] = np.nanmean(arr[:, start:end], axis=1)
        return out

    # Co-update time series (same vs cross context) with the Δr inset below
    def _draw_ts_block(fig, gs_slot, results, panel_type, doEmpSampling, show_title=True,
                        allow_break=True, sig_direction=1):
        title, same_key, diff_key, dep_key = panel_type
        _nrows = 2
        _hr = [2, 1]
        gs_block = GridSpecFromSubplotSpec(_nrows, 1, subplot_spec=gs_slot,
                                            height_ratios=_hr, hspace=0.5)

        # Mean ± SEM per layer (also used to decide whether to break the y-axis)
        x = None
        curves = []  # list of (mean_same, sem_same, mean_diff, sem_diff, ls) per layer
        for layer in layers_ts:
            res = results[layer]
            same, diff = res[same_key], res[diff_key]
            ls = ':' if layer == "hidden2" else '-'
            x = np.arange(1, np.sum(~np.isnan(same[0, :])) + 1)
            same, diff = same[:, :x[-1]], diff[:, :x[-1]]
            if not doEmpSampling:
                same, diff = _smooth_running(same, DENSE_SMOOTH_WIN), _smooth_running(diff, DENSE_SMOOTH_WIN)
            mean_same, sem_same = np.nanmean(same, axis=0), sem(same, axis=0, nan_policy="omit")
            mean_diff, sem_diff = np.nanmean(diff, axis=0), sem(diff, axis=0, nan_policy="omit")
            curves.append((mean_same, sem_same, mean_diff, sem_diff, ls))

        # If the same-context and cross-context traces lie in two separate y-ranges, the empty
        # gap between them is cut out with a broken y-axis.
        same_lo = min(np.nanmin(m - e) for m, e, _, _, _ in curves)
        same_hi = max(np.nanmax(m + e) for m, e, _, _, _ in curves)
        diff_lo = min(np.nanmin(m - e) for _, _, m, e, _ in curves)
        diff_hi = max(np.nanmax(m + e) for _, _, m, e, _ in curves)
        if not allow_break:
            _has_break = False  # no axis break (Fig. S6)
        elif same_lo > diff_hi:
            _top_lo, _top_hi, _bot_lo, _bot_hi = same_lo, same_hi, diff_lo, diff_hi
            _has_break = True
        elif diff_lo > same_hi:
            _top_lo, _top_hi, _bot_lo, _bot_hi = diff_lo, diff_hi, same_lo, same_hi
            _has_break = True
        else:
            _has_break = False  # the traces overlap

        if _has_break:
            _top_range = max(_top_hi - _top_lo, 1e-6)
            _bot_range = max(_bot_hi - _bot_lo, 1e-6)
            _gs_main = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_block[0],
                                                height_ratios=[_top_range, _bot_range], hspace=0.08)
            ax_main_top = fig.add_subplot(_gs_main[0])
            ax_main     = fig.add_subplot(_gs_main[1])
        else:
            ax_main_top = None
            ax_main     = fig.add_subplot(gs_block[0])
        ax_inset = fig.add_subplot(gs_block[1])

        if show_title:
            (ax_main_top or ax_main).set_title(title, fontsize=11)

        _plot_axes = [ax_main, ax_main_top] if _has_break else [ax_main]
        for mean_same, sem_same, mean_diff, sem_diff, ls in curves:
            for _ax in _plot_axes:
                _ax.plot(x, mean_same, color=base_same, lw=2, ls=ls)
                _ax.fill_between(x, mean_same - sem_same, mean_same + sem_same,
                                  color=base_same, alpha=0.2, linewidth=0)
                _ax.plot(x, mean_diff, color=base_diff, lw=2, ls=ls)
                _ax.fill_between(x, mean_diff - sem_diff, mean_diff + sem_diff,
                                  color=base_diff, alpha=0.2, linewidth=0)

        ax_main.set_ylabel("Co-update\n[r]", fontsize=9)
        ax_main.set_xticks([])
        ax_main.set_xlabel("")
        ax_main.set_xlim(1, x[-1])
        ax_main.tick_params(labelsize=7)
        ax_main.yaxis.set_major_formatter(_fmt_nolz)
        ax_main.spines["right"].set_visible(False)

        if _has_break:
            _pad = 0.12
            ax_main.set_ylim(_bot_lo - _pad * _bot_range, _bot_hi + _pad * _bot_range)
            ax_main_top.set_ylim(_top_lo - _pad * _top_range, _top_hi + _pad * _top_range)

            ax_main.spines["top"].set_visible(False)
            ax_main_top.spines["top"].set_visible(False)
            ax_main_top.spines["right"].set_visible(False)
            ax_main_top.spines["bottom"].set_visible(False)
            ax_main_top.set_xlim(1, x[-1])
            ax_main_top.set_xticks([])
            ax_main_top.set_xlabel("")
            ax_main_top.tick_params(labelsize=7)
            ax_main_top.yaxis.set_major_formatter(_fmt_nolz)

            # diagonal break marks on the left axis
            _d = 0.012
            _kw = dict(transform=ax_main_top.transAxes, color='k', clip_on=False, lw=1)
            ax_main_top.plot((-_d, +_d), (-_d, +_d), **_kw)
            _kw['transform'] = ax_main.transAxes
            ax_main.plot((-_d, +_d), (1 - _d, 1 + _d), **_kw)
        else:
            ax_main.axhline(0, ls="--", lw=0.8, color='k')
            ax_main.spines["top"].set_visible(False)
            ymin, ymax = ax_main.get_ylim()
            max_abs = max(abs(ymin), abs(ymax))
            ax_main.set_ylim(-max_abs, max_abs)

        # Δr inset
        _sig_by_layer = {}  # layer -> (sig_mask, linestyle), FDR-corrected, for the bars below
        _dep_lo, _dep_hi = [], []   # data extent, to pin the y-limits to the traces alone
        for layer in layers_ts:
            res = results[layer]
            dep = res[dep_key][:, :x[-1]]
            if not doEmpSampling:
                dep = _smooth_running(dep, DENSE_SMOOTH_WIN)
            ls = ':' if layer == "hidden2" else '-'
            mean_dep, sem_dep = np.nanmean(dep, axis=0), sem(dep, axis=0, nan_policy="omit")
            ax_inset.plot(x, mean_dep, color=base_dep, lw=1.8, ls=ls)
            ax_inset.fill_between(x, mean_dep - sem_dep, mean_dep + sem_dep,
                                   color=base_dep, alpha=0.2, linewidth=0)
            _dep_lo.append(np.nanmin(mean_dep - sem_dep))
            _dep_hi.append(np.nanmax(mean_dep + sem_dep))

            # Paired t-test of each time point vs the first one (before learning), FDR-corrected
            # (Benjamini-Hochberg) within each layer. Only effects in sig_direction are marked
            # (+1: above baseline, -1: below).
            _baseline = dep[:, 0]
            _pvals, _t_idx, _tvals = [], [], []
            for _t in range(1, x[-1]):
                _a, _b = dep[:, _t], _baseline
                _valid = np.isfinite(_a) & np.isfinite(_b)
                if _valid.sum() > 1:
                    _tstat, _p = ttest_rel(_a[_valid], _b[_valid])
                    _pvals.append(_p)
                    _t_idx.append(_t)
                    _tvals.append(_tstat)
            _sig_mask = np.zeros(x[-1], dtype=bool)
            if _pvals:
                _rej, _, _, _ = multipletests(_pvals, alpha=0.05, method='fdr_bh')
                _keep = _rej & (np.sign(_tvals) == sig_direction)
                _sig_mask[_t_idx] = _keep
            _sig_by_layer[layer] = (_sig_mask, ls)

        ax_inset.axhline(0, ls="--", lw=0.8, color='k')

        # y-limits from the traces (and 0); the significance bars are drawn above the axes
        _lo, _hi = min(min(_dep_lo), 0.0), max(max(_dep_hi), 0.0)
        _pad_i = 0.04 * max(_hi - _lo, 1e-6)
        ax_inset.set_ylim(_lo - _pad_i, _hi + _pad_i)

        # Significance bars above the panel: RNN layer on top (solid), FC layer below (dotted)
        _sig_trans = transforms.blended_transform_factory(ax_inset.transData, ax_inset.transAxes)
        _sig_ls = {'hidden2': (0, (1, 1.2))}  # fine dots for FC; RNN stays solid
        for _li, layer in enumerate(layers_ts):
            _sig_mask, _ = _sig_by_layer[layer]
            # Stack downward from above the axis top, with a clear gap between the two bars.
            _y_sig = 1.20 - _li * 0.11
            _edges = np.diff(np.concatenate(([0], _sig_mask.astype(int), [0])))
            for _s, _e in zip(np.where(_edges == 1)[0], np.where(_edges == -1)[0] - 1):
                ax_inset.plot([x[_s], x[_e]], [_y_sig, _y_sig], color='gold', lw=1.5,
                              ls=_sig_ls.get(layer, '-'), transform=_sig_trans,
                              zorder=5, clip_on=False)

        ax_inset.set_ylabel("Context-dependent\nco-update [$\\Delta r$]", fontsize=9)
        ax_inset.set_xlim(1, x[-1])
        ax_inset.tick_params(labelsize=7)
        ax_inset.yaxis.set_major_formatter(_fmt_nolz)
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)
        _x_ticks = np.arange(2, 21, 2) if doEmpSampling else np.arange(50, x[-1] + 1, 50)
        ax_inset.set_xticks(_x_ticks)
        ax_inset.set_xlabel("Learning segment #" if doEmpSampling else "Learning trial #",
                             fontsize=9, labelpad=10)

        _align_axes = [ax_main, ax_inset]
        fig.align_ylabels(_align_axes)

    # Mean co-update of each participant over the segments of each class (from coupdate_analysis_RNN.py)
    class_keys = {
        "delta_corr":        ("ctx_dep_coup_no_gen_mean",      "ctx_dep_coup_gen_mean",      "ctx_dep_coup_sat_mean"),
        "corr_same_context": ("same_context_coup_no_gen_mean", "same_context_coup_gen_mean", None),
        "corr_diff_context": ("diff_context_coup_no_gen_mean", "diff_context_coup_gen_mean", None)}

    def _compute_subj_means(results, layer, dv, conditions):
        res = results[layer]
        no_gen_key, gen_key, sat_key = class_keys[dv]
        subj_means = pd.DataFrame({"no_generalization": res[no_gen_key],
                                   "generalization": res[gen_key]}, index=subjects)
        if sat_key is not None:
            subj_means["saturated_learning"] = res[sat_key]
        subj_means.index.name = "sub"
        return subj_means[conditions]

    # Paired t-tests, both layers
    def _print_stats_both_layers(results, dv, conditions, col_title=None):
        _tag = f"{col_title} | " if col_title else ""
        for layer in ['RNN_hidden1', 'hidden2']:
            subj_means_l = _compute_subj_means(results, layer, dv, conditions)

            for (c1), (c2) in itertools.combinations(conditions, 2):
                v1, v2 = subj_means_l[c1], subj_means_l[c2]
                paired_l = pd.concat([v1, v2], axis=1).dropna()
                if len(paired_l) > 1:
                    diff_l = paired_l.iloc[:, 0].values - paired_l.iloc[:, 1].values
                    t, p = ttest_rel(paired_l.iloc[:, 0], paired_l.iloc[:, 1])
                    d = np.mean(diff_l) / np.std(diff_l, ddof=1)
                    print(f"  [paired]   {_tag}{layer} | {dv} | {c1} vs {c2}: "
                          f"t({len(paired_l) - 1})={t:.3f}, p={p:.4g}, d={d:.3f}")

    # Correlation of the mean context-dependent co-update with mean accuracy, both layers
    def _print_coupdate_acc_correlation(results, col_title=None):
        _tag = f"{col_title} | " if col_title else ""
        for layer in ['RNN_hidden1', 'hidden2']:
            res = results[layer]
            mean_coup = np.nanmean(res['context_dep_coup_all'], axis=1)
            acc = res['acc_learning']
            mask = np.isfinite(mean_coup) & np.isfinite(acc)
            if mask.sum() > 1:
                rho, p_s = spearmanr(mean_coup[mask], acc[mask])
                print(f"  [corr]     {_tag}{layer}: spearman(mean context-dep co-update, mean accuracy)"
                      f" = {rho:.3f}, p = {p_s:.4g}, n = {int(mask.sum())}")
                r, p = pearsonr(mean_coup[mask], acc[mask])
                print(f"  [corr]     {_tag}{layer}: pearson(mean context-dep co-update, mean accuracy)"
                      f" = {r:.3f}, p = {p:.4g}, n = {int(mask.sum())}")

    # Paired swarm panel
    def _draw_paired_panel(fig, gs_slot, results, dv, conditions, ylabel, tick_size, dot_size, col_title=None):
        layer = layer_for_panels
        layer_color = panel_colors[dv]

        print(f"[Figure7 stats] dv={dv}, conditions={conditions}")
        _print_stats_both_layers(results, dv, conditions, col_title=col_title)

        subj_means = _compute_subj_means(results, layer, dv, conditions)

        xpos = np.arange(len(conditions))
        df_plot = subj_means.reset_index().melt(
            id_vars="sub", value_vars=conditions, var_name="condition", value_name=dv)
        df_plot["x"] = df_plot["condition"].map({c: i for i, c in enumerate(conditions)})

        valid_vals = subj_means.values.flatten()
        valid_vals = valid_vals[~np.isnan(valid_vals)]

        # y-range from the no-generalization and generalization points; points outside it
        # (a few saturated ones) are cropped at the frame
        _range_ref_cols = [c for c in ("no_generalization", "generalization") if c in subj_means.columns]
        _range_ref_vals = (subj_means[_range_ref_cols].values.flatten() if len(_range_ref_cols) == 2
                            else valid_vals)
        _range_ref_vals = _range_ref_vals[~np.isnan(_range_ref_vals)]
        if len(_range_ref_vals):
            yr0 = np.ptp(_range_ref_vals) if np.ptp(_range_ref_vals) > 0 else 1e-6
            y_min = np.min(_range_ref_vals) - 0.06 * yr0
            y_max = np.max(_range_ref_vals) + 0.06 * yr0
        else:
            y_min, y_max = 0, 1

        ax = fig.add_subplot(gs_slot)

        for cond in conditions:
            sns.swarmplot(x="x", y=dv, data=df_plot[df_plot["condition"] == cond],
                          ax=ax, size=dot_size, color=layer_color)

        if connect_subjects:
            _conn_color = layer_color if dv != 'delta_corr' else 'k'
            for sub in subj_means.index:
                y_vals = subj_means.loc[sub].values
                for i in range(len(xpos) - 1):
                    if not (np.isnan(y_vals[i]) or np.isnan(y_vals[i + 1])):
                        ax.plot([xpos[i] + 0.3, xpos[i + 1] - 0.3],
                                [y_vals[i], y_vals[i + 1]], color=_conn_color, alpha=0.3)

        if doPlotMeanDifference:
            paired_full = subj_means.dropna()
            if len(paired_full) > 0:
                means = paired_full.mean().values
                for i in range(len(xpos) - 1):
                    ax.plot([xpos[i] + 0.3, xpos[i + 1] - 0.3],
                            [means[i], means[i + 1]], color="darkred", lw=2, zorder=3)

        ax.set_ylim(y_min, y_max)
        yr = y_max - y_min

        def _paired_stats(c1, c2, pp):
            # Paired t-test between two conditions, over the participants with both (pp).
            # Returns (t, p, d, n); p = None when there are fewer than 2 participants.
            if len(pp) > 1:
                diff = pp.iloc[:, 0].values - pp.iloc[:, 1].values
                t, p = ttest_rel(pp.iloc[:, 0], pp.iloc[:, 1])
                d = np.mean(diff) / np.std(diff, ddof=1)
                return t, p, d, len(pp)
            return None, None, None, 0

        if len(conditions) == 3:
            # Only test gen vs no_gen and gen vs saturated; all brackets at same height
            _pairs_3 = [
                (conditions.index("no_generalization"), "no_generalization",
                 conditions.index("generalization"),    "generalization"),
                (conditions.index("generalization"),    "generalization",
                 conditions.index("saturated_learning"), "saturated_learning"),
            ]
            y_line = y_max + 0.08 * yr
            for i1, c1, i2, c2 in _pairs_3:
                v1, v2 = subj_means[c1], subj_means[c2]
                paired_pair = pd.concat([v1, v2], axis=1).dropna()
                t, p, d, _npair = _paired_stats(c1, c2, paired_pair)
                if p is not None and np.round(p, 2) <= 0.1:
                    ax.plot([i1, i2], [y_line, y_line], color="k", lw=1, clip_on=False)
                    ax.plot([i1, i1], [y_line - 0.03*yr, y_line], color="k", lw=1, clip_on=False)
                    ax.plot([i2, i2], [y_line - 0.03*yr, y_line], color="k", lw=1, clip_on=False)
                    ax.text((i1 + i2) / 2, y_line + 0.001 * yr, significance_stars(p),
                            ha="center", va="bottom", fontsize=12, clip_on=False)
                    ax.text((i1 + i2) / 2, y_line - 0.025 * yr, format_p_value(p),
                            ha="center", va="top", fontsize=6, clip_on=False)
        else:
            sig_count = 0
            for (i1, c1), (i2, c2) in itertools.combinations(enumerate(conditions), 2):
                v1, v2 = subj_means[c1], subj_means[c2]
                paired_pair = pd.concat([v1, v2], axis=1).dropna()
                t, p, d, _npair = _paired_stats(c1, c2, paired_pair)
                if p is not None and np.round(p, 2) <= 0.1:
                    y_line = y_max + (0.08 + sig_count * 0.20) * yr
                    ax.plot([i1, i2], [y_line, y_line], color="k", lw=1, clip_on=False)
                    ax.plot([i1, i1], [y_line - 0.03*yr, y_line], color="k", lw=1, clip_on=False)
                    ax.plot([i2, i2], [y_line - 0.03*yr, y_line], color="k", lw=1, clip_on=False)
                    ax.text((i1 + i2) / 2, y_line + 0.001 * yr, significance_stars(p),
                            ha="center", va="bottom", fontsize=12, clip_on=False)
                    ax.text((i1 + i2) / 2, y_line - 0.025 * yr, format_p_value(p),
                            ha="center", va="top", fontsize=6, clip_on=False)
                    sig_count += 1

        if y_min < 0 < y_max:
            ax.axhline(0, ls="--", lw=0.8, color="k")

        ax.set_xticks(xpos)
        ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=9)
        ax.set_xlabel("", fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.yaxis.set_major_formatter(_fmt_nolz)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=tick_size)
        sns.despine(ax=ax)

    #%% Figure 7: empirical | dense sampling
    _fig_h      = 7.0
    _top_hratio = 2.75
    fig7_main = plt.figure(figsize=(7.2, _fig_h))
    outer_main = GridSpec(3, 2, figure=fig7_main,
                           height_ratios=[_top_hratio, 1.5, 1.5], hspace=0.65, wspace=0.7)

    for col_idx, (col_title, res_col, doEmp) in enumerate(columns):
        fig7_main.text(0.28 + col_idx * 0.5, 0.99, col_title, ha='center', fontsize=11, fontweight='bold')

        # Row 1: co-update time series (context feature) and Δr inset
        _draw_ts_block(fig7_main, outer_main[0, col_idx], res_col, context_panel_type, doEmp,
                        show_title=False)

        print(f"\n[Figure7 stats] === {col_title} ===")
        _print_coupdate_acc_correlation(res_col, col_title=col_title)

        # Row 2: context-dependent co-update by generalization class
        _draw_paired_panel(fig7_main, outer_main[1, col_idx], res_col, "delta_corr", conditions_3,
                            "Context-dependent\nco-update [$\\Delta r$]", 9, 3, col_title=col_title)

        # Row 3: cross-context and within-context co-update by generalization class
        gs_row3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_main[2, col_idx], wspace=0.65)
        _draw_paired_panel(fig7_main, gs_row3[0], res_col, "corr_diff_context", conditions_2,
                            "Cross-contexts\nco-update [r]", 7, 2.5, col_title=col_title)
        _draw_paired_panel(fig7_main, gs_row3[1], res_col, "corr_same_context", conditions_2,
                            "Within-context\nco-update [r]", 7, 2.5, col_title=col_title)

    _add_ts_legend(fig7_main, 0.955)

    fig7_main.savefig(os.path.join(save_dir, f"Figure7_main_{simulation}{_align_suffix}{_proc_suffix}.pdf"),
                       dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig7_main)

    #%% Fig. S6: irrelevant context feature, empirical | dense sampling
    _supp_fig_h = _fig_h * (_top_hratio / (_top_hratio + 1.5 + 1.5))  # same height as row 1 of Figure 7
    fig7_supp = plt.figure(figsize=(7.2, _supp_fig_h))
    # same space for title and legend (in inches) as in Figure 7
    _supp_top = 1 - 0.84 / _supp_fig_h
    outer_supp = GridSpec(1, 2, figure=fig7_supp, wspace=0.35, top=_supp_top)

    for col_idx, (col_title, res_col, doEmp) in enumerate(columns):
        fig7_supp.text(0.28 + col_idx * 0.5, 0.97, col_title, ha='center', fontsize=11, fontweight='bold')
        _draw_ts_block(fig7_supp, outer_supp[0, col_idx], res_col, irrel_panel_type, doEmp,
                        allow_break=False, sig_direction=-1)

    _add_ts_legend(fig7_supp, (_supp_top + 1) / 2)

    fig7_supp.savefig(os.path.join(save_dir, f"Figure7_supp_irrelevant_{simulation}{_align_suffix}{_proc_suffix}.pdf"),
                       dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig7_supp)

    # free memory
    del results_emp, results_dense
    gc.collect()

