class PheWAS_Pool:
    """
    Class for performing PheWAS
    ======================================================================================================
    phecode_counts: Pandas Dataframe of Phecodes
    covariates: Pandas Dataframe of covariates to include in the analysis
    indep_var: String indicating the column in covariates that is the independent variable of interest
    CDR_version: String indicating CDR version
    phecode_process: list for phecodes to process
    min_cases: minimum number of cases for an individual phenotype to be analyzed
    cores: if not "", then specify number of cores to use in the analysis
    indep_alt: alternative independent variable for non-linear associations
    """

    def __init__(
        self,
        phecode_counts,
        covariates,
        indep_var_of_interest="",
        CDR_version="R2019Q4R3",
        phecode_process="all",
        reg=0,
        min_cases=100,
        var_names=[
            "AF",
            "white",
            "Asian",
            "male",
            "age_at_last_event",
            "ehr_length",
            "code_cnt",
        ],
        gender_var_names=[
            "AF",
            "white",
            "Asian",
            "male",
            "age_at_last_event",
            "ehr_length",
            "code_cnt",
        ],
        show_res=False,
    ):
        print(
            "~~~~~~~~~~~~~~~        Creating PheWAS AOU Object           ~~~~~~~~~~~~~~~~~~~~~"
        )
        # create instance attributes
        self.var_interest = indep_var_of_interest
        self.return_dict = []
        self.return_dict_alt = []
        self.CDR_version = CDR_version
        # self.demo_patients_phecodes=pd.merge(covariates,phecode_counts,on="person_id") #NA as newly updated
        self.pp = phecode_counts
        self.ppc = phecode_counts[phecode_counts["count"] >= 2].groupby("phecode")
        self.cov = covariates
        if phecode_process == "all":
            self.phecode_list = sorted(phecode_counts["phecode"].unique().tolist())
        else:
            self.phecode_list = sorted(phecode_process)  # a list, not np array
        self.show_res = show_res
        self.var_names = var_names
        self.var_names = list(np.append(np.array([self.var_interest]), self.var_names))
        self.gender_var_names = gender_var_names
        self.gender_var_names = list(
            np.append(np.array([self.var_interest]), self.gender_var_names)
        )
        self.remove_dup = list(np.append(np.array(["person_id"]), self.var_names))
        self.min_cases = min_cases
        self.reg = reg

    def runPheLogit(self, phecodes):  # phecodes should be pre-sorted
        cov = self.cov
        ppc = self.ppc
        pp = self.pp
        px = ICD9_exclude
        gr = gender_restriction
        print("Total: ", len(phecodes))
        nError = 0
        controls_pre = pd.DataFrame()
        code_pre = ["0.1"]
        gender_pre = -1
        for idx, phecode in enumerate(phecodes):
            error = "Other Error"
            try:
                include = ppc.get_group(phecode)
                cases = cov[cov.person_id.isin(include["person_id"])]
                g_tmp = gr[gr["phecode"] == phecode].iloc[0]
                gender = -1
                if g_tmp["male_only"] == True:
                    gender = 0
                if g_tmp["female_only"] == True:
                    gender = 1
                if (
                    gender == 0
                ):  # gender_var_names should not contain both male and female
                    var_names = self.gender_var_names
                    cases = cases[cases["male"] == 1]
                elif gender == 1:
                    var_names = self.gender_var_names
                    cases = cases[cases["female"] == 1]
                else:
                    var_names = self.var_names

                cases = cases[self.remove_dup].drop_duplicates()[var_names]
                if cases.shape[0] >= self.min_cases:
                    phecode_x = (
                        px[px["code"] == phecode]["exclusion_criteria"]
                        .unique()
                        .tolist()
                    )
                    phecode_x.append(phecode)
                    if set(code_pre) != set(phecode_x) or gender_pre != gender:
                        exclude = pp[pp["phecode"].isin(phecode_x)].person_id
                        if gender == 0:
                            control_ids = cov[
                                (cov.person_id.isin(exclude) == False) & cov.male == 1
                            ].person_id
                        elif gender == 1:
                            control_ids = cov[
                                (cov.person_id.isin(exclude) == False) & cov.female == 1
                            ].person_id
                        else:
                            control_ids = cov[
                                cov.person_id.isin(exclude) == False
                            ].person_id
                        control = cov[cov.person_id.isin(control_ids)]
                        control = control[self.remove_dup].drop_duplicates()[var_names]
                        code_pre = phecode_x
                        gender_pre = gender
                        controls_pre = control
                    else:
                        control = controls_pre
                    ############################################################################################
                    ## Perform Logistic regression
                    ## Now run through the logit function from stats models
                    ############################################################################################
                    y = [1] * cases.shape[0] + [0] * control.shape[0]
                    regressors = pd.concat([cases, control])
                    regressors = sm.tools.add_constant(regressors)

                    if self.reg == 1:
                        logit = sm.OLS(y, regressors, missing="drop")
                    elif self.reg == 2:
                        logit = sm.GLS(y, regressors, missing="drop")
                    else:
                        logit = sm.Logit(y, regressors, missing="drop")
                    result = logit.fit(disp=False)

                    if self.show_res == True:
                        print(result.summary())
                    else:
                        pass
                    if idx % 100 == 0:
                        print(idx, phecode, ": ", cases.shape[0], control.shape[0])
                    results_as_html = result.summary().tables[0].as_html()
                    converged = pd.read_html(results_as_html)[0].iloc[5, 1]
                    results_as_html = result.summary().tables[1].as_html()
                    res = pd.read_html(results_as_html, header=0, index_col=0)[0]

                    p_value = result.pvalues[self.var_interest]
                    beta_ind = result.params[self.var_interest]
                    conf_int_1 = res.loc[self.var_interest]["[0.025"]
                    conf_int_2 = res.loc[self.var_interest]["0.975]"]

                    self.return_dict.append(
                        [
                            phecode,
                            cases.shape[0],
                            control.shape[0],
                            p_value,
                            beta_ind,
                            conf_int_1,
                            conf_int_2,
                            converged,
                        ]
                    )
                else:
                    code_pre = ["0.1"]
                    # error = "Error in Phecode: "+str(phecode)+ ": Number of cases less than minimum of "+str(self.min_cases)
                del [control, cases, regressors]
            except:
                nError += 1
                # print(error)
        print("nError ", nError)

    # now define function for running the phewas
    def run(self):
        self.runPheLogit(self.phecode_list)
        logit_Phecode_results = self.return_dict

        # TODO: instead of creating dataframe, just write the csv files that come from the dataframe
        logit_Phecode_results = pd.DataFrame(logit_Phecode_results)
        # print(logit_Phecode_results.shape)
        if logit_Phecode_results.shape[0] > 0:
            logit_Phecode_results.columns = [
                "phecode",
                "cases",
                "control",
                "p_value",
                "beta_ind",
                "conf_int_1",
                "conf_int_2",
                "converged",
            ]
            logit_Phecode_results["code_val"] = logit_Phecode_results["phecode"]
            logit_Phecode_results["neg_p_log_10"] = -np.log10(
                logit_Phecode_results["p_value"]
            )
            logit_Phecode_results = pd.merge(phecode_info, logit_Phecode_results)
            # now save logit phecode as attribute
            self.logit_Phecode_results = logit_Phecode_results

    def Manhattan_Plot_Plus(self, group="all"):
        """
        Method for plotting Manhattan Plot
        ======================================================================================================
        group: list of groups to display (e.g. neoplasms)
        """
        PheWAS_results_ehr = self.logit_Phecode_results

        PheWAS_results_ehr.loc[
            PheWAS_results_ehr["color"] == "darkorange1", "color"
        ] = "orange"
        PheWAS_results_ehr.loc[
            PheWAS_results_ehr["color"] == "darkseagreen4", "color"
        ] = "darkgreen"
        PheWAS_results_ehr.loc[
            PheWAS_results_ehr["color"] == "coral4", "color"
        ] = "coral"
        PheWAS_results_ehr.loc[
            PheWAS_results_ehr["color"] == "chartreuse4", "color"
        ] = "chartreuse"
        PheWAS_results_ehr.loc[
            PheWAS_results_ehr["color"] == "royalblue4", "color"
        ] = "royalblue"
        PheWAS_results_ehr.loc[
            PheWAS_results_ehr["color"] == "gray50", "color"
        ] = "gray"

        # subset to particular group

        if group != "all":
            PheWAS_results_ehr = PheWAS_results_ehr[
                PheWAS_results_ehr["group"] == group
            ]

        fig, ax = plt.subplots(figsize=(15, 8))
        benf_corr = 0.05 / phecodes.PheCode.unique().shape[0]
        pos_beta = PheWAS_results_ehr[PheWAS_results_ehr["beta_ind"] >= 0]
        neg_beta = PheWAS_results_ehr[PheWAS_results_ehr["beta_ind"] < 0]

        ax.scatter(
            pos_beta["code_val"],
            pos_beta["neg_p_log_10"],
            c=pos_beta["color"],
            marker="^",
        )
        ax.scatter(
            neg_beta["code_val"],
            neg_beta["neg_p_log_10"],
            c=neg_beta["color"],
            marker="v",
        )
        ax.hlines(
            -np.log10(0.05),
            0,
            PheWAS_results_ehr["code_val"].max() + 1,
            colors="r",
            label="0.05",
        )
        ax.hlines(
            -np.log10(benf_corr),
            0,
            PheWAS_results_ehr["code_val"].max() + 1,
            colors="g",
            label="Bonferroni Threshold (0.05/1847)",
        )
        PheWas_ticks = (
            PheWAS_results_ehr[["phecode", "group"]]
            .groupby("group", as_index=False)
            .mean()
        )

        # reshape the final plot to just fit the phecodes in the subgroup
        plt.xlim(
            PheWAS_results_ehr["phecode"].min(), PheWAS_results_ehr["phecode"].max()
        )
        plt.xticks(
            PheWas_ticks["phecode"], PheWas_ticks["group"], rotation=45, ha="right"
        )
        pos_beta_top = (
            pos_beta[pos_beta["p_value"] < benf_corr]
            .sort_values("neg_p_log_10", ascending=False)
            .iloc[:15,][["code_val", "neg_p_log_10", "description"]]
        )
        # Drop infs
        #
        pos_beta_top = pos_beta_top[~np.isinf(pos_beta_top["neg_p_log_10"])]
        neg_beta_top = (
            neg_beta[neg_beta["p_value"] < benf_corr]
            .sort_values("neg_p_log_10", ascending=False)
            .iloc[:10,][["code_val", "neg_p_log_10", "description"]]
        )
        ## drop infs
        neg_beta_top = neg_beta_top[~np.isinf(neg_beta_top["neg_p_log_10"])]

        for i, row in pos_beta_top.iterrows():
            ax.annotate(row["code_val"], (row["code_val"], row["neg_p_log_10"]))
        for i, row in neg_beta_top.iterrows():
            ax.annotate(row["code_val"], (row["code_val"], row["neg_p_log_10"]))
        # assign top pos and neg to self
        self.pos_beta_top = pos_beta_top
        self.neg_beta_top = neg_beta_top
        from matplotlib.lines import Line2D

        # add legend elements
        legend_elements = [
            Line2D([0], [0], color="g", lw=4, label="Bonferroni Correction"),
            Line2D([0], [0], color="r", lw=4, label="Nominal Significance Level"),
            Line2D(
                [0],
                [0],
                marker="v",
                label="Protective Effect",
                markerfacecolor="b",
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                label="Non-Protective Effect",
                markerfacecolor="b",
                markersize=15,
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        ax.set_ylabel(r"$-\log_{10}$(p-value)")
