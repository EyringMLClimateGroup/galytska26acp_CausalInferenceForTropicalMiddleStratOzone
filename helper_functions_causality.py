from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import copy
import pandas as pd

from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.causal_effects import CausalEffects
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.models import Models, LinearMediation


def fix_graph(graph):
    """
    Standardize edge labels in a graph array and warn about ambiguous edges.
    """
    graph = np.array(graph, dtype=str)

    # Replace 'o->' with '-->'
    graph[graph == "o->"] = "-->"

    # Replace '<-o' with '<--'
    graph[graph == "<-o"] = "<--"

    # Check for problematic values and print warnings
    if np.any(graph == "<->"):
        print("Attention.. <-> is in the graph. Fix this manually.")

    if np.any(graph == "x-x"):
        print("Attention.. x--x is in the graph. Fix this manually.")

    return graph


def add_link(graph, src, tgt, tau):
    """Add source (src) -> target (tgt) at lag tau. Mirror only for tau=0."""
    graph[src, tgt, tau] = "-->"
    # add mirrored links for lag 0
    if tau == 0:
        graph[tgt, src, tau] = "<--"

    return graph


def get_variable_index(var_list, var):
    """
    Return the index of a variable in a list, raising an error if not found.
    """
    try:
        return var_list.index(var)
    except ValueError:
        raise ValueError(f"Variable '{var}' not found in var_list.")


def convert_dict_to_val_matrix(var_list, dict_with_wright_coeffs, par):
    """
    Convert a nested dictionary of Wright coefficients into a list of
    val_matrices.

    For each mask in the dictionary, builds a (N, N, tau_max+1) matrix.
    At tau=0, the matrix is symmetrized.
    """

    # par - 'beta' or 'beta_ci'
    val_matrix_list = []
    N = len(var_list)
    for mask in dict_with_wright_coeffs.keys():
        tau_max = max(dict_with_wright_coeffs[mask].keys())
        val_matrix = np.zeros((N, N, tau_max + 1))

        # Loop over the dictionary and fill the matrix
        for tau, links_dict in dict_with_wright_coeffs[mask].items():
            for (var1, _, var2), value in links_dict[par].items():
                i = get_variable_index(var_list, var1)
                j = get_variable_index(var_list, var2)
                val_matrix[i, j, tau] = value
            # Mirror the values for tau = 0
        val_matrix[:, :, 0] = (
            val_matrix[:, :, 0]
            + val_matrix[:, :, 0].T
            - np.diag(val_matrix[:, :, 0].diagonal())
        )

        val_matrix_list.append(val_matrix)

    return val_matrix_list


mask_group = {
    "No_mask": "no_mask",
    "east": "east",
    "west": "west",
}


def position_of_nodes(list_with_vars):
    """
    Hardcoded x/y positions for a set of variable combinations.
    """
    if len(list_with_vars) == 2 and list_with_vars == ["w*", "N$_2$O"]:
        node_pos = {"x": np.array([2.5, 1.1]), "y": np.array([0.1, 3.5])}

    elif len(list_with_vars) == 3 and list_with_vars == ["w*", "N$_2$O", "NO$_2$"]:
        node_pos = {"x": np.array([2.5, 1.1, 0.5]), "y": np.array([0.1, 3.5, 7.0])}

    elif len(list_with_vars) == 3 and list_with_vars == ["N$_2$O", "NO$_2$", "O$_3$"]:
        node_pos = {"x": np.array([1.1, 0.5, 5.5]), "y": np.array([3.5, 7.0, 9.5])}

    elif len(list_with_vars) == 4 and list_with_vars == [
        "w*",
        "N$_2$O",
        "NO$_2$",
        "O$_3$",
    ]:
        node_pos = {
            "x": np.array([2.5, 1.1, 0.5, 5.5]),
            "y": np.array([0.1, 3.5, 7.0, 9.5]),
        }

    elif len(list_with_vars) == 4 and list_with_vars == [
        "N$_2$O",
        "NO$_2$",
        "O$_3$",
        "T",
    ]:
        node_pos = {
            "x": np.array([1.1, 0.5, 5.5, 4.5]),
            "y": np.array([3.5, 7.0, 9.5, 5.0]),
        }

    elif len(list_with_vars) == 5 and list_with_vars == [
        "w*",
        "N$_2$O",
        "NO$_2$",
        "O$_3$",
        "T",
    ]:
        node_pos = {
            "x": np.array([2.5, 1.1, 0.5, 5.5, 4.5]),
            "y": np.array([0.3, 3.5, 7.0, 9.5, 5.0]),
        }

    return node_pos


def lpcmci_sensitivity(
    dataframe,
    Rparcorr,
    var_names,
    tau_max,
    tau_min,
    n_preliminary_iterations,
    link_assumptions=None,
    save_fig=None,
):
    """
    Run LPCMCI across five significance levels (0.01, 0.02, 0.05, 0.1, 0.2)
    and plot the resulting graphs.
    """
    lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=Rparcorr, verbosity=0)
    Rresults001 = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        n_preliminary_iterations=n_preliminary_iterations,
        link_assumptions=link_assumptions,
        pc_alpha=0.01,
    )
    Rresults002 = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        n_preliminary_iterations=n_preliminary_iterations,
        link_assumptions=link_assumptions,
        pc_alpha=0.02,
    )
    Rresults005 = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        n_preliminary_iterations=n_preliminary_iterations,
        link_assumptions=link_assumptions,
        pc_alpha=0.05,
    )
    Rresults01 = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        n_preliminary_iterations=n_preliminary_iterations,
        link_assumptions=link_assumptions,
        pc_alpha=0.1,
    )
    Rresults02 = lpcmci.run_lpcmci(
        tau_min=tau_min,
        tau_max=tau_max,
        n_preliminary_iterations=n_preliminary_iterations,
        link_assumptions=link_assumptions,
        pc_alpha=0.2,
    )

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))
    axes[0].set_title("pc_alpha=0.2")
    tp.plot_graph(
        val_matrix=Rresults02["val_matrix"],
        graph=Rresults02["graph"],
        var_names=var_names,
        fig_ax=(fig, axes[0]),
        node_pos=position_of_nodes(var_names),
        node_size=1.2,
        show_colorbar=False,
    )

    axes[1].set_title("pc_alpha=0.1")
    tp.plot_graph(
        val_matrix=Rresults01["val_matrix"],
        graph=Rresults01["graph"],
        var_names=var_names,
        fig_ax=(fig, axes[1]),
        node_pos=position_of_nodes(var_names),
        node_size=1.2,
        show_colorbar=False,
    )

    axes[2].set_title("pc_alpha=0.05")
    tp.plot_graph(
        val_matrix=Rresults005["val_matrix"],
        graph=Rresults005["graph"],
        var_names=var_names,
        fig_ax=(fig, axes[2]),
        node_pos=position_of_nodes(var_names),
        node_size=1.2,
        show_colorbar=False,
    )

    axes[3].set_title("pc_alpha=0.02")
    tp.plot_graph(
        val_matrix=Rresults002["val_matrix"],
        graph=Rresults002["graph"],
        var_names=var_names,
        fig_ax=(fig, axes[3]),
        node_pos=position_of_nodes(var_names),
        node_size=1.2,
        show_colorbar=False,
    )

    axes[4].set_title("pc_alpha=0.01")
    tp.plot_graph(
        val_matrix=Rresults001["val_matrix"],
        graph=Rresults001["graph"],
        var_names=var_names,
        fig_ax=(fig, axes[4]),
        node_pos=position_of_nodes(var_names),
        node_size=1.2,
        show_colorbar=False,
    )

    if save_fig is not None:
        plt.savefig(save_fig, dpi=130, bbox_inches="tight")
    plt.show()

    return Rresults02, Rresults01, Rresults005, Rresults002, Rresults001


def remove_nan_integer_flag(array, missing_data_flag=999.0):
    """
    Replaces integer flag by NaNs in a numpy array
    Args:
    ----
    array: numpy array with integer flag for missing values
    missing_data_flag: value of integer flag
    """

    array_w_nan = copy.deepcopy(array)
    array_w_nan[array_w_nan == missing_data_flag] = np.nan

    return array_w_nan


def get_valid_range(dataframe, causal_var, show_graph=True):
    """
    Returns [5th - 95th] percentile interval of a given variable in a
    tigramite dataframe.

    Args:
    ----
    dataframe: tigramite dataframe
    causal_var: given variable for calculation of interval
    show_graph: True/False, will show a histogram of causal_var
    """
    a = remove_nan_integer_flag(dataframe.values[0][:, causal_var[0][0]])

    if show_graph:
        plt.hist(a)

    a_valid_range_min = round(np.nanpercentile(a, 5), 3)
    a_valid_range_max = round(np.nanpercentile(a, 95), 3)

    return [a_valid_range_min, a_valid_range_max]


def calculate_wright_coeffs(
    df,
    source,
    list_of_X=[0, 1, 2, 3],
    list_of_Y=[0, 1, 2, 3],
    list_of_lags=[0, -1, -2, -3, -4, -5],
    graph_list=None,
    var_names=None,
    boot_samples=500,
    conf_level=0.90,
    boot_blocklength=1,
    seed=None,
):
    """
    Calculates Wright coefficients and bootstrap confidence intervals
    for multiple datasets, variable pairs, and lags.
    Only stores results where the CI does not include zero.

    Returns:
        test_dict: nested dict with 'beta' and 'beta_ci' for each
        cause-effect pair.
    """
    dataframes = {key: df[source][key]["dataframe"] for key in df[source].keys()}
    list_of_causes = list_of_X
    list_of_effects = list_of_Y
    cause_lag_list = list_of_lags
    test_dict = {}

    for id, (key, d) in enumerate(dataframes.items()):

        print(id, key)

        test_dict[key] = {}
        print(f"Processing key: {key}")
        graph = graph_list[id]

        for cause_lag in cause_lag_list:
            test_dict[key][abs(cause_lag)] = {
                "beta": {},
                "beta_ci": {},
                "beta_boot_samples": {},
            }

            for cause_var in list_of_causes:
                for effect_var in list_of_effects:
                    X = [(cause_var, cause_lag)]
                    Y = [(effect_var, 0)]

                    # Skip if cause and effect variables overlap
                    if any(x == y for x in X for y in Y):
                        print(
                            f"Skipping due to overlap: Cause {var_names[X[0][0]]}"
                            f"and Effect {var_names[Y[0][0]]}"
                        )
                        continue

                    # Get valid intervention range
                    valid_range_X = get_valid_range(d, X, show_graph=False)
                    intervention_data1 = valid_range_X[0] * np.ones((1, len(X)))
                    intervention_data2 = valid_range_X[1] * np.ones((1, len(X)))

                    # Create causal effects object
                    causal_effects = CausalEffects(
                        graph,
                        graph_type="stationary_dag",
                        X=X,
                        Y=Y,
                        S=None,
                        hidden_variables=None,
                        verbosity=0,
                    )

                    # ---- Estimate causal effects----
                    causal_effects.fit_wright_effect(
                        d, mediation="direct", mask_type="y"
                    )
                    y1 = causal_effects.predict_wright_effect(
                        intervention_data=intervention_data1
                    )
                    y2 = causal_effects.predict_wright_effect(
                        intervention_data=intervention_data2
                    )
                    beta = (y1 - y2) / (valid_range_X[0] - valid_range_X[1])

                    # ---- Bootstrap CIs ----
                    # Check if there is X--> Y link
                    if getattr(causal_effects, "no_causal_path", False):
                        # No path: can't bootstrap, set CI as zero
                        ci = np.array([[0.0], [0.0]])
                        beta_boot_samples = np.zeros((boot_samples, 1))

                    else:
                        causal_effects.fit_bootstrap_of(
                            method="fit_wright_effect",
                            method_args={
                                "dataframe": d,
                                "mediation": "direct",
                                "mask_type": "y",
                            },
                            boot_samples=boot_samples,
                            boot_blocklength=boot_blocklength,
                            seed=seed,
                        )

                        # Predict bootstrap results for both intervention points
                        intervention_data = np.vstack(
                            [intervention_data1, intervention_data2]
                        )
                        boot_preds, _ = causal_effects.predict_bootstrap_of(
                            method="predict_wright_effect",
                            method_args={"intervention_data": intervention_data},
                            conf_lev=conf_level,
                            return_individual_bootstrap_results=True,
                        )

                        delta_x = valid_range_X[0] - valid_range_X[1]
                        beta_boot_samples = (
                            boot_preds[:, 0, :] - boot_preds[:, 1, :]
                        ) / delta_x

                        lower = np.percentile(
                            beta_boot_samples, (1 - conf_level) / 2 * 100, axis=0
                        )
                        upper = np.percentile(
                            beta_boot_samples, (1 + conf_level) / 2 * 100, axis=0
                        )
                        ci = np.stack([lower, upper], axis=0)

                    # ---- Significance check ----
                    if not (ci[0] > 0).all() and not (ci[1] < 0).all():
                        # CI includes zero --> skip
                        continue

                    # Store results
                    name_tuple = (var_names[cause_var], "--->", var_names[effect_var])
                    test_dict[key][abs(cause_lag)]["beta"][name_tuple] = beta
                    test_dict[key][abs(cause_lag)]["beta_ci"][name_tuple] = ci

    return test_dict


def get_links_coeffs_from_graph(dataframe, graph, tau_max=1):
    """
    Returns linear estimates of link coefficients of a given graph,
    calculated from a given dataframe.
    Args:
    ----
    dataframe: tigramite dataframe
    graph: numpy array causal graph
    tau_max: maximum time lag that will be considered
    """
    parents = toys.dag_to_links(graph)
    # ensure that lag info is int, note np.array or so
    for key, values in parents.items():
        parents[key] = [(var, int(lag)) for var, lag in values]
    model = Models(
        dataframe=dataframe,
        model=LinearRegression(),
        data_transform=None,
        mask_type="y",
        verbosity=0,
    )

    fit_res_dict = {}

    # Loop over the range of j from 0 to len(parents) - 1
    for j in range(len(parents)):

        # Call the model's get_general_fitted_model function for each j
        fit_res = model.get_general_fitted_model(
            # Y is based on j
            Y=[(j, 0)],
            # X is the list of parents for j
            X=list(parents[j]),
            # Z is empty
            Z=[],
            # No conditions specified
            conditions=None,
            # tau_max is provided by user
            tau_max=tau_max,
            # Using 'tau_max' as cut_off
            cut_off="tau_max",
            # Return the fitted result without data
            return_data=False,
        )

        # Store the result in the dictionary using j as the key
        fit_res_dict[j] = fit_res["model"].coef_

    links_coeffs = {}
    # Identity lin_f function: no transformation applied (linear relationship)
    lin_f = lambda x: x
    # Iterate through each item in the parents dictionary
    for key, parent_tuples in parents.items():
        # Get the corresponding coefficients and flatten them (1D array)
        coeffs = fit_res_dict[key].flatten()

        # Combine parent tuples, coefficients, and the lambda function
        combined = [
            (parent, coeff, lin_f) for parent, coeff in zip(parent_tuples, coeffs)
        ]

        # Add the result to the dictionary
        links_coeffs[key] = combined

    return links_coeffs


def temporal_evolution(
    dataframe,
    graph,
    list_x=[0],
    list_y=[1, 2, 3],
    mediator=None,
    blocked_mediator=None,
    tau_max=100,
    bootstrap=True,
    mask_names=None,
    save=None,
    boot_blocklength="cube_root",
    n_boots=500,
    verbose=False,
):
    """
    Returns linear estimates (Wright path method) of temporal evolution of the
    effect of X on Y, calculated from a given dataframe, with(out) mediation,
    with(out) bootstrap confidence intervals, with/out masks

    Args:
    ----
    dataframe: tigramite dataframe
    graph: numpy array causal graph
    list_x: list of cause-variables to loop over (integer)
    list_y: list of effect-variables to loop over (integer)
    mediator: integer that designates a mediator. The mediated effect will be
    calculated (as opposed to total effect)
    blocked_mediator: Effect that is not mediated by blocked_mediator will be
    returned
    tau_max: maximum time lag
    bootstrap: True/False, whether to compute the bootstrap ci
    save: directory where result csv will be save if not None
    boot_blocklength: argument from tigramite bootstrap function
    n_boots: number of bootstrap iterations
    """

    lags = range(tau_max)

    links_coeffs = get_links_coeffs_from_graph(dataframe, graph, tau_max=2)
    true_parents = toys._get_true_parent_neighbor_dict(links_coeffs)

    # if dataframe is masked, will loop over mask and ~mask
    if (dataframe.mask is None) or np.all(dataframe.mask[0] == 0):
        print("Data is not masked")
        dataframes = [dataframe]
        mask_type_names = ["not_masked"]
    else:
        print("Data is masked")
        dataframes = [dataframe]
        mask_type_names = mask_names if mask_names is not None else ["masked"]

    columns = [
        "lag",
        "effect",
        "direct_effect",
        "conf_low",
        "conf_high",
        "mediator",
        "path",
    ]

    all_results = pd.DataFrame(data=None, columns=columns)

    for d_id, d in enumerate(dataframes):
        # Initialize dataframe object, specify time axis and variable names
        med = LinearMediation(dataframe=d, mask_type="y")
        med.fit_model(all_parents=true_parents, tau_max=tau_max)
        if bootstrap:
            med.fit_model_bootstrap(
                boot_samples=n_boots, boot_blocklength=boot_blocklength
            )

        # loop over paths to estimate
        for i in list_x:
            for j in list_y:
                path = f"{i}_to_{j}"
                lagged_effects = []
                direct_effects = []
                conf_intervs = []
                for l in lags:
                    if mediator is None:
                        lagged_effect = med.get_ce(
                            i=i,
                            tau=-l,
                            j=j,
                        )
                        direct_effect = med.get_coeff(
                            i=i,
                            tau=-l,
                            j=j,
                        )
                        if bootstrap:
                            conf_interv = med.get_bootstrap_of(
                                function="get_ce",
                                function_args={"i": i, "tau": -l, "j": j},
                                conf_lev=0.9,
                            )
                        else:
                            conf_interv = [np.nan, np.nan]

                    if mediator is not None:
                        if blocked_mediator is not None:
                            lagged_effect = med.get_conditional_mce(
                                i=i, tau=-l, j=j, k=mediator, notk=blocked_mediator
                            )
                            if bootstrap:
                                conf_interv = med.get_bootstrap_of(
                                    "get_conditional_mce",
                                    {
                                        "i": i,
                                        "tau": -l,
                                        "k": mediator,
                                        "notk": blocked_mediator,
                                        "j": j,
                                    },
                                    conf_lev=0.9,
                                )
                            else:
                                conf_interv = [np.nan, np.nan]

                        else:
                            lagged_effect = med.get_mce(i=i, tau=-l, j=j, k=mediator)
                            if bootstrap:
                                conf_interv = med.get_bootstrap_of(
                                    function="get_mce",
                                    function_args={
                                        "i": i,
                                        "tau": -l,
                                        "j": j,
                                        "k": mediator,
                                    },
                                    conf_lev=0.9,
                                )
                            else:
                                conf_interv = [np.nan, np.nan]

                    lagged_effects.append(lagged_effect)
                    conf_intervs.append(conf_interv)
                    if mediator is None:
                        direct_effects.append(direct_effect)
                    else:
                        direct_effects.append(np.nan)
                # group results
                results_this_path = pd.DataFrame(
                    data=np.vstack(
                        [
                            np.array(lags),
                            np.array(lagged_effects),
                            np.array(direct_effects),
                            np.array(conf_intervs)[:, 0],
                            np.array(conf_intervs)[:, 1],
                        ]
                    ).transpose(),
                    columns=["lag", "effect", "direct_effect", "conf_low", "conf_high"],
                )

                results_this_path["mediator"] = mediator
                results_this_path["path"] = path
                if mediator is None:
                    results_this_path["direct effect"] = np.array(direct_effects)
                results_this_path["mask_type"] = mask_type_names[d_id]

                all_results = pd.concat(
                    [all_results, results_this_path], axis=0, ignore_index=True
                )

        if verbose:
            print(f"CE for dataset {d_id} done")

    # save results
    if save is not None:
        all_results.to_csv(save + f"temporal_dvpt_all_paths_linear_mediation.csv")

    return all_results


def plot_temporal_evolution(new_dictionary, var_names, mediator, shared_legend=True):
    """Plot the temporal evolution of causal links across QBO regimes."""
    # Map dictionary keys to desired legend labels
    period_mapping = {
        "No_mask": "Full period 2004-2021",
        "east": "Eaterlies",
        "west": "Westerlies",
    }

    # Get unique subplots based on the paths in the first period's first source
    first_period = list(new_dictionary.keys())[0]
    first_source = list(new_dictionary[first_period].keys())[0]

    # Assuming all dataframes have the same subplots
    subplots = np.unique(new_dictionary[first_period][first_source].path)

    n_plots = len(subplots)
    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))
    f, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    y_plot = "effect"
    # Customize markers for different data sources
    markers = {
        "OBS": "o",
        "TOMCAT": "^",
    }
    # Track the index of the plot to use axes correctly
    plot_idx = 0
    # Dictionary to store handles for the legend
    handles_dict = {}
    # Dictionary to store colors for each period based on the first source ("OBS")
    period_colors = {}

    # Create custom handles for the legend
    obs_line_handle = plt.Line2D(
        [0], [0], color="black", linewidth=2, linestyle="-", label="OBS (solid line)"
    )
    tomcat_line_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        linewidth=2,
        linestyle="--",
        label="MODEL (dashed line)",
    )

    # Create transparent marker handles with black edge color
    obs_marker_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=10,
        label="OBS (circle marker)",
    )
    tomcat_marker_handle = plt.Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=10,
        label="MODEL (triangle marker)",
    )

    for p in subplots:
        if p[0] == p[-1]:
            continue
        print(p[0], p[-1])
        ax = axes[plot_idx]

        for period_key in period_mapping.keys():
            # Access the results for each period across sources
            period_results = new_dictionary[period_key]
            # Get the mapped label
            period_label = period_mapping[period_key]

            for source, df in period_results.items():
                # Filter the data for the current path `p`
                results_this_path = df[df.path == p]

                line_width = 1
                # Initialize line_color
                line_color = None

                plot_data = results_this_path[results_this_path[y_plot] != 0]
                if not plot_data.empty:

                    # If the period color is not yet stored,
                    # plot with the source OBS and store its color
                    if source == "OBS":
                        # Plot and get the color for this period
                        lineplot = sns.lineplot(
                            x="lag",
                            y=y_plot,
                            data=plot_data,
                            ax=ax,
                            label=period_label,
                            linewidth=line_width,
                            linestyle="-",
                            legend=False,
                        )
                        # Save the color used by OBS
                        line_color = lineplot.get_lines()[-1].get_color()
                        # Store the color for later use
                        period_colors[period_key] = line_color

                        # Only add to handles_dict if the label hasn't been added yet
                        if period_label not in handles_dict:
                            # Store the last line handle for this label
                            handles_dict[period_label] = lineplot.get_lines()[-1]

                    elif source == "TOMCAT":
                        # Use the stored color for TOMCAT and set the line style to dashed
                        line_color = period_colors.get(period_key)
                        sns.lineplot(
                            x="lag",
                            y=y_plot,
                            data=plot_data,
                            ax=ax,
                            label=None,
                            linewidth=line_width,
                            linestyle="--",
                            color=line_color,
                            legend=False,
                        )

                    # Plot confidence intervals
                    ax.fill_between(
                        x=plot_data["lag"],
                        y1=plot_data["conf_low"],
                        y2=plot_data["conf_high"],
                        color=line_color,
                        alpha=0.15,
                    )

                first_element = results_this_path[
                    results_this_path["direct_effect"] != 0
                ]

                if not first_element.empty:
                    # Get the first occurrence with direct effect
                    first_lag = first_element.iloc[0]
                    sns.scatterplot(
                        x="lag",
                        y="direct_effect",
                        # Convert Series to DataFrame for plotting
                        data=first_lag.to_frame().T,
                        ax=ax,
                        # Get the marker for the current source
                        marker=markers.get(source, "o"),
                        # Increase marker size for better visibility
                        s=130,
                        color=line_color,
                        # Add a black edge to the markers
                        edgecolor="black",
                        linewidth=0.5,
                        legend=False,
                    )

        # Set plot labels
        ax.set_ylabel(f"Total effect {var_names[int(p[0])]} to {var_names[int(p[-1])]}")
        plot_idx += 1
        max_lag = (
            int(plot_data["lag"].max())
            if not plot_data.empty
            else int(results_this_path["lag"].max())
        )
        ax.set_xticks(np.arange(0, max_lag + 1, 1))
        ax.set_xlim(-0.3, max_lag + 0.3)
        ax.set_xlabel("Lag (months)")
        if mediator is not None:
            ax.set_title(f"Mediated by {var_names[mediator]}")

    # Hide any unused subplots
    for j in range(plot_idx, n_cols * n_rows):
        f.delaxes(axes[j])

    # Create a shared legend outside the subplots for solid lines only
    if shared_legend:
        # Get the stored handles
        legend_handles = list(handles_dict.values())
        # Get the corresponding labels
        legend_labels = list(handles_dict.keys())
        f.legend(
            legend_handles
            + [
                obs_line_handle,
                tomcat_line_handle,
                obs_marker_handle,
                tomcat_marker_handle,
            ],
            legend_labels
            + ["Observations", "TOMCAT CTM", "Observations", "TOMCAT CTM"],
            loc="upper center",
            bbox_to_anchor=(0.3, 0.04),
            ncol=len(period_mapping) + 4,
        )
    f.tight_layout()
    plt.show()
