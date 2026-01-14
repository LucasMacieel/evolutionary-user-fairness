from data_loader import DataLoader
from utils.tools import create_logger, evaluation_methods
import pandas as pd
import pulp
import os


class UGF(object):
    def __init__(
        self,
        data_loader,
        k,
        eval_metric_list,
        fairness_metric="f1",
        epsilon="auto",
        logger=None,
        model_name="",
        group_name="",
    ):
        """
        Train fairness model
        :param data_loader: Dataloader object
        :param k: k for top-K number of items to be selected from the entire list
        :param eval_metric_list: a list contains all the metrics to report
        :param fairness_metric: a string, the metric used for fairness constraint, default='f1'
        :param epsilon: the upper bound for difference between groups.
                        Use 'auto' to calculate as half of original UGF gap (paper methodology)
        :param logger: logger for logging info
        """
        self.data_loader = data_loader
        self.dataset_name = data_loader.path.split("/")[-1]
        self.k = k
        self.eval_metric_list = eval_metric_list
        self.fairness_metric = fairness_metric
        self._epsilon_input = epsilon  # Store original input
        self.epsilon = None  # Will be set in train() if 'auto'
        self.model_name = model_name
        self.group_name = group_name
        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

    @staticmethod
    def _check_df_format(df):
        """
        check if the input dataframe contains all the necessary columns
        :return: None
        """
        expected_columns = ["uid", "iid", "score", "label", "q"]
        for c in expected_columns:
            if c not in df.columns:
                raise KeyError("Missing column " + c)

    @staticmethod
    def _build_fairness_optimizer(group_df_list, k, metric, name="UGF"):
        """
        Use PuLP to build fairness optimizer
        :param group_df_list: a list contains dataframes from two groups
        :param k: an integer for the length of top-K list
        :param metric: the metric string for fairness constraint. e.g. 'f1', 'recall', 'precision'
        :param name: a string which is the name of this optimizer
        :return: the PuLP problem, a list of Qi*Si for obj function, a list of two group metric, dict of variables
        """
        # Create a new maximization problem
        prob = pulp.LpProblem(name, pulp.LpMaximize)
        var_score_list = []  # store the Qi * Si
        metric_list = []  # store the averaged metric scores for two groups
        all_vars = {}  # store all variables by name for result extraction

        # Create variables
        for df in group_df_list:
            df_group = df.groupby("uid")
            tmp_metric_list = []  # store the metric calculation for each user in current user group
            tmp_var_score_list = []  # store var * score for object function use
            for uid, group in df_group:
                tmp_var_list = []  # store variables for sum(Qi) == k use
                tmp_var_label_list = []  # store var * label for recall calculation use
                score_list = group["score"].tolist()
                label_list = group["label"].tolist()
                item_list = group["iid"].tolist()

                for i in range(len(item_list)):
                    var_name = (
                        str(uid) + "_" + str(item_list[i])
                    )  # variable name is "uid_iid"
                    v = pulp.LpVariable(var_name, cat="Binary")
                    all_vars[var_name] = v
                    tmp_var_list.append(v)
                    tmp_var_score_list.append(score_list[i] * v)
                    tmp_var_label_list.append(label_list[i] * v)

                # Add first constraint: Sum(Qi)==k
                prob += pulp.lpSum(tmp_var_list) == k

                # calculate the corresponding measures
                if group["label"].sum() == 0:
                    continue
                if metric == "recall":
                    tmp_metric_list.append(
                        pulp.lpSum(tmp_var_label_list) / group["label"].sum()
                    )
                elif metric == "precision":
                    tmp_metric_list.append(pulp.lpSum(tmp_var_label_list) / k)
                elif metric == "f1":
                    f1 = 2 * pulp.lpSum(tmp_var_label_list) / (group["label"].sum() + k)
                    tmp_metric_list.append(f1)
                else:
                    raise ValueError("Unknown metric for optimizer building.")

            metric_list.append(pulp.lpSum(tmp_metric_list) / len(tmp_metric_list))
            var_score_list.extend(tmp_var_score_list)

        return prob, var_score_list, metric_list, all_vars

    @staticmethod
    def _format_result(all_vars, df):
        """
        format the PuLP results to dataframe.
        :param all_vars: dictionary of variable name to LpVariable
        :param df: the pandas dataframe to add the optimized results into
        :return: None
        """
        # Build results list first (much faster than row-by-row updates)
        results = []
        for var_name, v in all_vars.items():
            v_s = var_name.split("_")
            uid = int(v_s[0])
            iid = int(v_s[1])
            results.append({"uid": uid, "iid": iid, "q_new": int(pulp.value(v))})

        # Create DataFrame from results and merge (vectorized, much faster)
        results_df = pd.DataFrame(results)
        df.drop(columns=["q"], inplace=True)
        merged = df.merge(results_df, on=["uid", "iid"], how="left")
        merged.rename(columns={"q_new": "q"}, inplace=True)

        # Update original df in place
        df["q"] = merged["q"].values

    def _print_metrics(self, df, metrics, message="metric scores"):
        """
        Print out evaluation scores
        :param df: the dataframe contains the data for evaluation
        :param metrics: a list, contains the metrics to report
        :param message: a string, for print message
        :return: None
        """
        results = evaluation_methods(df, metrics=metrics)
        r_string = ""
        for i in range(len(metrics)):
            r_string = r_string + metrics[i] + "=" + "{:.4f}".format(results[i]) + " "
        print(message + ": " + r_string)
        # write the message into the log file
        self.logger.info(message + ": " + r_string)
        return results  # Return results for UGF gap calculation

    def _calculate_ugf_gap(self, group1_df, group2_df, metric):
        """
        Calculate the original UGF gap between two groups.
        :param group1_df: advantaged group dataframe
        :param group2_df: disadvantaged group dataframe
        :param metric: metric string (e.g. 'f1@10')
        :return: absolute difference between group metrics
        """
        g1_results = evaluation_methods(group1_df, metrics=[metric])
        g2_results = evaluation_methods(group2_df, metrics=[metric])
        gap = abs(g1_results[0] - g2_results[0])
        return gap

    def train(self):
        """
        Train fairness model
        """
        # Prepare data
        all_df = self.data_loader.rank_df.copy(
            deep=True
        )  # the dataframe with entire test data
        self._check_df_format(all_df)  # check the dataframe format
        group_df_list = [
            self.data_loader.g1_df.copy(deep=True),
            self.data_loader.g2_df.copy(deep=True),
        ]  # group 1 (active), group 2 (inactive)

        # Print original evaluation results
        self.logger.info(
            "Model:{} | Dataset:{} | Group:{} |  Epsilon={} | K={} | Fairness_metric={}".format(
                self.model_name,
                self.dataset_name,
                self.group_name,
                self.epsilon,
                self.k,
                self.fairness_metric,
            )
        )
        self._print_metrics(
            all_df,
            self.eval_metric_list,
            "Before optimization overall scores           ",
        )
        self._print_metrics(
            group_df_list[0],
            self.eval_metric_list,
            "Before optimization group 1 (active) scores  ",
        )
        self._print_metrics(
            group_df_list[1],
            self.eval_metric_list,
            "Before optimization group 2 (inactive) scores",
        )

        # Calculate and log original UGF
        fairness_metric_k = self.fairness_metric + "@" + str(self.k)
        original_ugf = self._calculate_ugf_gap(
            group_df_list[0], group_df_list[1], fairness_metric_k
        )
        print(
            f"Before optimization UGF ({fairness_metric_k}): {original_ugf:.4f} ({original_ugf * 100:.2f}%)"
        )
        self.logger.info(
            f"Before optimization UGF ({fairness_metric_k}): {original_ugf:.4f} ({original_ugf * 100:.2f}%)"
        )

        # Calculate epsilon dynamically if set to 'auto' (paper methodology)
        if self._epsilon_input == "auto":
            self.epsilon = original_ugf / 4  # One quarter of original UGF gap
            print("\nDynamic epsilon calculation (paper methodology):")
            print(
                f"  Original UGF gap ({fairness_metric_k}): {original_ugf:.4f} ({original_ugf * 100:.2f}%)"
            )
            print(
                f"  Epsilon (half of gap): {self.epsilon:.4f} ({self.epsilon * 100:.2f}%)"
            )
            self.logger.info(
                f"Dynamic epsilon: original_gap={original_ugf:.4f}, epsilon={self.epsilon:.4f}"
            )
        else:
            self.epsilon = self._epsilon_input

        # build optimizer
        prob, var_score_list, metric_list, all_vars = self._build_fairness_optimizer(
            group_df_list, self.k, metric=self.fairness_metric, name="UGF_f1"
        )

        # |group_1_metric - group_2_metric| <= epsilon
        prob += metric_list[0] - metric_list[1] <= self.epsilon
        prob += metric_list[1] - metric_list[0] <= self.epsilon

        # Set objective function
        prob += pulp.lpSum(var_score_list)

        # Optimize model with timing
        import time

        print("Solving optimization problem with PuLP (HiGHS solver)...")
        start_time = time.time()
        prob.solve(pulp.HiGHS_CMD(msg=1))
        end_time = time.time()
        cpu_time = end_time - start_time

        print(f"\nStatus: {pulp.LpStatus[prob.status]}")
        print(f"CPU time: {cpu_time:.2f} seconds")
        self.logger.info(f"Solver status: {pulp.LpStatus[prob.status]}")
        self.logger.info(f"CPU time: {cpu_time:.2f} seconds")

        # Format the output results and update q column of the dataframe
        self._format_result(all_vars, all_df)
        group_df_list[0].drop(columns=["q"], inplace=True)
        group_df_list[0] = pd.merge(
            group_df_list[0], all_df, on=["uid", "iid", "score", "label"], how="left"
        )
        group_df_list[1].drop(columns=["q"], inplace=True)
        group_df_list[1] = pd.merge(
            group_df_list[1], all_df, on=["uid", "iid", "score", "label"], how="left"
        )

        # Print updated evaluation results
        self._print_metrics(
            all_df,
            self.eval_metric_list,
            "After optimization overall metric scores     ",
        )
        self._print_metrics(
            group_df_list[0],
            self.eval_metric_list,
            "After optimization group 1 (active) scores   ",
        )
        self._print_metrics(
            group_df_list[1],
            self.eval_metric_list,
            "After optimization group 2 (inactive) scores ",
        )

        # Calculate and log optimized UGF
        optimized_ugf = self._calculate_ugf_gap(
            group_df_list[0], group_df_list[1], fairness_metric_k
        )
        print(
            f"After optimization UGF ({fairness_metric_k}): {optimized_ugf:.4f} ({optimized_ugf * 100:.2f}%)"
        )
        self.logger.info(
            f"After optimization UGF ({fairness_metric_k}): {optimized_ugf:.4f} ({optimized_ugf * 100:.2f}%)"
        )

        # Log UGF improvement
        ugf_reduction = original_ugf - optimized_ugf
        ugf_reduction_pct = (
            (ugf_reduction / original_ugf) * 100 if original_ugf > 0 else 0
        )
        print(
            f"UGF reduction: {ugf_reduction:.4f} ({ugf_reduction_pct:.1f}% improvement)"
        )
        self.logger.info(
            f"UGF reduction: {ugf_reduction:.4f} ({ugf_reduction_pct:.1f}% improvement)"
        )

        self.logger.info("\n")

        # Capture final metrics for return
        final_metrics = evaluation_methods(all_df, self.eval_metric_list)

        return {
            "final_ugf": optimized_ugf,
            "final_metrics": final_metrics,
            "constraint_satisfied": optimized_ugf <= self.epsilon + 1e-6,
            "original_ugf": original_ugf,
            "epsilon": self.epsilon,
        }


if __name__ == "__main__":
    """
    Comprehensive evaluation: runs all dataset/model/grouping combinations
    to replicate the paper's experimental setup.
    """
    import csv
    from datetime import datetime

    ############### Configuration ###########
    epsilon = "auto"  # Dynamic epsilon (paper methodology)
    dataset_folder = "../dataset"

    # All available datasets
    datasets = ["5Beauty-rand", "5Grocery-rand", "5Health-rand"]

    # All available models (must have corresponding *_rank.csv files)
    models = ["NCF", "biasedMF"]

    # Grouping methods: (group_name, group_1_suffix, group_2_suffix)
    grouping_methods = [
        (
            "0.05_count",
            "0.05_count_active_test_ratings.txt",
            "0.05_count_inactive_test_ratings.txt",
        ),
        (
            "sum_0.05",
            "sum_0.05_price_active_test_ratings.txt",
            "sum_0.05_price_inactive_test_ratings.txt",
        ),
        (
            "max_0.05",
            "max_0.05_price_active_test_ratings.txt",
            "max_0.05_price_inactive_test_ratings.txt",
        ),
    ]

    metrics = ["ndcg", "f1"]
    topK = ["10"]
    metrics_list = [metric + "@" + k for metric in metrics for k in topK]

    # Results collection
    all_results = []

    # Create results directory
    results_base_dir = "../results"
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)

    ############### Run all combinations ###########
    total_runs = len(datasets) * len(models) * len(grouping_methods)
    current_run = 0

    for dataset_name in datasets:
        for model_name in models:
            for group_name, group_1_file, group_2_file in grouping_methods:
                current_run += 1
                print("\n" + "=" * 80)
                print(
                    f"RUN {current_run}/{total_runs}: {dataset_name} | {model_name} | {group_name}"
                )
                print("=" * 80)

                try:
                    # Setup paths
                    data_path = os.path.join(dataset_folder, dataset_name)
                    rank_file = model_name + "_rank.csv"

                    # Check if rank file exists
                    if not os.path.exists(os.path.join(data_path, rank_file)):
                        print(f"  SKIPPED: {rank_file} not found")
                        continue

                    # Setup logging
                    logger_dir = os.path.join(results_base_dir, model_name)
                    if not os.path.exists(logger_dir):
                        os.makedirs(logger_dir)
                    logger_file = (
                        f"{model_name}_{dataset_name}_{group_name}_reRank_result.log"
                    )
                    logger_path = os.path.join(logger_dir, logger_file)

                    # Load data
                    dl = DataLoader(
                        data_path,
                        rank_file=rank_file,
                        group_1_file=group_1_file,
                        group_2_file=group_2_file,
                    )

                    logger = create_logger(
                        name=f"logger_{dataset_name}_{model_name}_{group_name}",
                        path=logger_path,
                    )

                    # Run UGF model
                    UGF_model = UGF(
                        dl,
                        k=10,
                        eval_metric_list=metrics_list,
                        fairness_metric="f1",
                        epsilon=epsilon,
                        logger=logger,
                        model_name=model_name,
                        group_name=group_name,
                    )

                    # Get before/after results
                    results = UGF_model.train()

                    # Store results for summary
                    all_results.append(
                        {
                            "Dataset": dataset_name.replace("5", "").replace(
                                "-rand", ""
                            ),
                            "Model": model_name,
                            "Grouping": group_name,
                            "Epsilon": UGF_model.epsilon,
                            "Status": "Completed",
                        }
                    )

                    print("  ✓ Completed successfully")

                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    all_results.append(
                        {
                            "Dataset": dataset_name.replace("5", "").replace(
                                "-rand", ""
                            ),
                            "Model": model_name,
                            "Grouping": group_name,
                            "Epsilon": "N/A",
                            "Status": f"Error: {str(e)}",
                        }
                    )

    ############### Summary ###########
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)

    # Print summary table
    print(
        f"\n{'Dataset':<12} {'Model':<12} {'Grouping':<12} {'Epsilon':<10} {'Status':<20}"
    )
    print("-" * 70)
    for r in all_results:
        eps_str = (
            f"{r['Epsilon']:.4f}" if isinstance(r["Epsilon"], float) else r["Epsilon"]
        )
        print(
            f"{r['Dataset']:<12} {r['Model']:<12} {r['Grouping']:<12} {eps_str:<10} {r['Status']:<20}"
        )

    # Save summary to CSV
    summary_file = os.path.join(
        results_base_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Dataset", "Model", "Grouping", "Epsilon", "Status"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSummary saved to: {summary_file}")
    print(f"Individual logs saved to: {results_base_dir}/<model_name>/")
    print("\nAll experiments completed!")
