import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(['nature'])

import pickle

import os
import sys

import seaborn as sns


from scipy.stats import sem, gaussian_kde, mannwhitneyu, wilcoxon, spearmanr, pearsonr, normaltest, ttest_ind

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

information_flow = ["ECin", "DG", "CA3", "CA1", "ECout"]

class AbstractAnalysis(object):

    def __init__(self, args):
        super(AbstractAnalysis, self).__init__()
        self.xp_id = args.experiment_id

    @property
    def xp_path(self):
        return "./logs/"+self.xp_id

    @property
    def processed_results_path(self):
        return self.xp_path+"/processed_results.df"

    @property
    def figures_path(self):
        return self.xp_path+"/figures"

    @property
    def simulations_path(self):
        return self.xp_path+"/simulations"

    def prepare_df(self):

        # If dataframe file already created, load it
        if os.path.isfile(self.processed_results_path):
            self.results = pickle.load(open(self.processed_results_path, 'rb'))
            print("loading")

        else:
            simulation_dirs = [self.simulations_path+'/'+simulation_dir for simulation_dir in os.listdir(self.simulations_path)]
            self.results = [] # temporary list to store the data to be stored in the dataframe

            for d in simulation_dirs:

                params = pickle.load(open(d+"/params.pickle",'rb'))
                data = pickle.load(open(d+"/results.pickle",'rb'))

                if params["recurrent_label"] == "CA3" or True:

                    data = utils.flatten_dict(data)

                    data = {k:v for k,v in data.items() if k not in ["pretraining_corrects", "training_corrects", 'pretraining_R_losses', 'training_R_losses'] and "losses" not in k}

                    params = {k:v for k,v in params.items() if k!="xp_cls"}

                    for k in list(data.keys()):

                        if 'corrects' in k:
                            # print(data[k].shape)
                            data['_'.join(['mean',k])] = np.mean(data[k]) if len(data[k])>0 else None

                    # Add params to data
                    for k,v in params.items():
                        data[k] = v



                    if params["modulations"] is None:
                        data["mechanisms"] = -1
                        data["target_labels"] = (-1,)
                        data["n_targets"] = 0
                        for target_label in constants.MODULAR_LAYERS:
                            data[target_label] = 0
                    else:
                        assert (np.array([params["modulations"][i].mechanism for i in range(len(params["modulations"]))])==params["modulations"][0].mechanism).all() # For now, only one mechanism is used at a time
                        data["mechanisms"] = params["modulations"][0].mechanism # For now, only one mechanism is used at a time
                        data["target_labels"] = tuple([params["modulations"][i].target_label for i in range(len(params["modulations"]))])
                        data["n_targets"] = len(data["target_labels"])

                        for target_label in constants.MODULAR_LAYERS:
                            data[target_label] = int(target_label in data["target_labels"])

                    if data["n_targets"] == 1:
                        data["target_recurrent_delta"] = information_flow.index(data["target_labels"][0]) - information_flow.index(data["recurrent_label"])
                    else:
                        data["target_recurrent_delta"] = None

                    data = {k:v for k,v in data.items() if k not in ["test_corrects", "mechanisms", "env", "args", "modulations", "rng"]}


                    self.results.append(data)

            print(len(self.results))

            self.results = pd.DataFrame(self.results)

            # Save
            with open(self.processed_results_path, "wb") as f:
                pickle.dump(self.results, f)
    
    @property
    def default_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==True) & 
            (self.results['n_pretraining_sessions']==1) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def allrecurrent_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['bio_count']==True) &
            (self.results['msp']==True) & 
            (self.results['n_pretraining_sessions']==1) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def manyshots_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==True) & 
            (self.results['n_pretraining_sessions']==20) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def manyshotsnomhn_results(self):
        return self.results[
            (self.results['use_mhn']==0) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==True) & 
            (self.results['n_pretraining_sessions']==20) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def manyshotsnomsp_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==False) & 
            (self.results['n_pretraining_sessions']==20) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def manyshotsnomspnomhn_results(self):
        return self.results[
            (self.results['use_mhn']==0) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==False) & 
            (self.results['n_pretraining_sessions']==20) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def nomhn_results(self):
        return self.results[
            (self.results['use_mhn']==0) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==True) & 
            (self.results['n_pretraining_sessions']==1) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def nomsp_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==False) & 
            (self.results['n_pretraining_sessions']==1) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def CA1rec_nomsp_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['recurrent_label']=="CA1") & 
            (self.results['bio_count']==True) &
            (self.results['msp']==False) & 
            (self.results['n_pretraining_sessions']==1) & 
            (self.results['n_training_sessions']==1)
        ]
    
    @property
    def nobiocount_results(self):
        return self.results[
            (self.results['use_mhn']==1) & 
            (self.results['recurrent_label']=="CA3") & 
            (self.results['bio_count']==False) &
            (self.results['msp']==True) & 
            (self.results['n_pretraining_sessions']==1) & 
            (self.results['n_training_sessions']==1)
        ]

    def __call__(self):
        os.makedirs(self.figures_path, exist_ok=True)
        self.prepare_df()
        self.analyse()

    def analyse(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    def target_labels_plot(self, df):

        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use(['nature'])

        flierprops = dict(marker='o', markersize=2)
        df = df[df['n_targets'] < 2]
        df["target_labels"] = df["target_labels"].astype(str)
        order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)", "(-1,)"]
        x='target_labels'
        y='mean_test_corrects'

        ax = sns.boxplot(
            data=df,
            x=x,
            y=y,
            showfliers = True,
            flierprops=flierprops,
            order=order,
            boxprops={"facecolor": (.4, .6, .8, .25)})


        xlim = ax.get_xlim()
        ax.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
        ax.set_xlim(xlim)
        ax.set_xlabel("Target")
        ax.set_ylabel("Performance (% correct)")
        ax.set_xticklabels([t.get_text()[2:-3] if t.get_text()!="(-1,)" else "None" for t in ax.get_xticklabels()])
        ax.set_ylim(40,102)

        fig = ax.get_figure()
        fig.tight_layout()

        return fig, ax
    
    def target_labels_pointplot(self, df):

        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use(['nature'])

        recurrent_label = df['recurrent_label'].iloc[0]
        assert np.all(df["recurrent_label"].to_numpy()==recurrent_label)
        use_mhn = df['use_mhn'].iloc[0]
        assert np.all(df["use_mhn"].to_numpy()==use_mhn)

        # df = df[df["seed"]<100]

        flierprops = dict(marker='o', markersize=2)
        df = df[df['n_targets'] < 2]
        df["target_labels"] = df["target_labels"].astype(str)
        order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)", "(-1,)"]
        order_clean = ["ECin", "DG", "CA3", "CA1", "ECout", "None"]
        df["target_labels"] = df["target_labels"].replace({label:order.index(label) for label in order})
        df["target_labels"] = df["target_labels"].astype(int)

        print(df["target_labels"])
        x='target_labels'
        y='mean_test_corrects'

        my_pal = {str(i): "#e64b35" if order_clean.index(recurrent_label) == i and use_mhn else "#8491b4" for i in range(len(order_clean))}
        print(my_pal)
        ax = sns.pointplot(
            data=df,
            x=y,
            y=x,
            errorbar=("ci", 99),
            order=range(len(order)),
            orient="h",
            linestyle="none",
            # capsize=.4,
            palette=my_pal,
        )


        ylim = ax.get_ylim()
        ax.plot([50 for i in range(2)], [-10,10], linestyle='--', color='k', alpha=.2, zorder=-1)
        ax.set_ylim(ylim)
        ax.set_ylabel("Target")
        ax.set_xlabel("Performance (% correct)")
        ax.set_yticklabels(order_clean)
        ax.set_xlim(45,70)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.tick_params(top=False,
               bottom=True,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)


        fig = ax.get_figure()
        fig.tight_layout()

        return fig, ax
    
    def target_labels_joyplot(self, df):

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        plt.style.use(['nature'])

        recurrent_label = df['recurrent_label'].iloc[0]
        assert np.all(df["recurrent_label"].to_numpy()==recurrent_label)
        use_mhn = df['use_mhn'].iloc[0]
        assert np.all(df["use_mhn"].to_numpy()==use_mhn)

        df = df[df['n_targets'] < 2]
        df["target_labels"] = df["target_labels"].astype(str)
        order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)", "(-1,)"]
        order_clean = ["ECin", "DG", "CA3", "CA1", "ECout", "None"]

        x='target_labels'
        y='mean_test_corrects'

        # Initialize the FacetGrid object
        my_pal = {order[i]: "#e64b35" if order_clean.index(recurrent_label) == i and use_mhn else "#8491b4" for i in range(len(order_clean))}
        g = sns.FacetGrid(df, row=x, hue=x, hue_order=order, row_order=order, palette=my_pal, aspect=15, height=.5)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, y,
            bw_adjust=.5, clip_on=False,
            fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, y, clip_on=False, color="w", lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.set_xlim(30,90)
            ax.text(0, .2, label[2:-3] if label!="(-1,)" else "None", size=7, family="sans-serif", color="black",
                    ha="left", va="center", transform=ax.transAxes)


        g.map(label, x)

        ax = plt.gca()
        fig = ax.get_figure()

        

        ax.set_xlabel("Performance (% correct)")
        ax.set_ylabel("Target")

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.5)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.spines[['bottom']].set_visible(True)
        ax.tick_params(top=False,
               bottom=True,
               left=False,
               right=False,
               labelleft=True,
               labelbottom=True)
        ax.xaxis.set_tick_params(width=.8, size=3.5)


        plt.show()
        return fig, ax
    
    def mhn_vs_nomhn_plot(self):
        
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use(['nature'])

        flierprops = dict(marker='o', markersize=2)

        df = pd.concat([self.default_results, self.nomhn_results, self.manyshots_results, self.manyshotsnomhn_results])
        df = df[df["n_targets"]==1]


        my_pal = {0:"#8491b4", 1:"#e64b35"}
        ax = sns.boxplot(
            data=df,
            x="n_pretraining_sessions",
            y="mean_test_corrects",
            hue="use_mhn",
            showfliers = True,
            flierprops=flierprops,
            order=[20,1],
            palette=my_pal,
            )
        
        xlim = ax.get_xlim()
        ax.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
        ax.set_xlim(xlim)
        ax.set_xlabel("")
        ax.set_ylabel("Performance (% correct)")
        ax.set_xticks([0,1],["Many-shot", "One-shot"])
        ax.set_ylim(20,100)

        ax.spines[['right', 'top', 'bottom']].set_visible(False)
        ax.tick_params(top=False,
               bottom=False,
               left=True,
               right=False,
               labelleft=True,
               labelbottom=True)


        handles, previous_labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles=handles, labels=["None", "CA3"], title="Memory", fontsize='small', fancybox=True)

        fig = ax.get_figure()
        fig.tight_layout()

        

        return fig, ax
    
    def target_recurrent_delta_plot(self):
        
        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use(['nature'])

        flierprops = dict(marker='o', markersize=2)

        df = self.default_results
        df = df[df["n_targets"]==1]
        df["target_recurrent_delta"] = df["target_recurrent_delta"].astype(int)

        print(df.dtypes)



        my_pal = {"-2":"#f7c2bb", "-1":"#ee8577", "0":"#e64b35", "1":"#8491b4", "2":"#8491b4"}
        ax = sns.boxplot(
            data=df,
            x="target_recurrent_delta",
            y="mean_test_corrects",
            showfliers = True,
            flierprops=flierprops,
            # order=[1,-1,0],
            palette=my_pal,
            )
        
        xlim = ax.get_xlim()
        ax.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
        ax.set_xlim(xlim)
        ax.set_xlabel("")
        ax.set_ylabel("Performance (% correct)")
        ax.set_ylim(20,100)

        ax.spines[['right', 'top', 'bottom']].set_visible(False)
        ax.tick_params(top=False,
               bottom=False,
               left=True,
               right=False,
               labelleft=True,
               labelbottom=True)


        fig = ax.get_figure()
        fig.tight_layout()

        

        return fig, ax
    
    def all_targets_plot(self, recurrent_label):

        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.style.use(['nature'])

        flierprops = dict(marker='o', markersize=2)

        df = self.allrecurrent_results
        df = df[df["n_targets"] > 0]
        df["target_labels"] = df["target_labels"].astype(str)
        df = df[df["recurrent_label"]==recurrent_label]
        order = list(df.groupby(by=["target_labels"])["mean_test_corrects"].quantile(.25).sort_values(ascending=False).index)

        fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios':[3,2]})

        sns.boxplot(
            data=df,
            x="target_labels",
            y="mean_test_corrects",
            ax=ax[0],
            showfliers = True,
            flierprops=flierprops,
            order=order,
            # boxprops={"facecolor": (.4, .6, .8, .25)},
            boxprops={"facecolor": (1, 1, 1, 1)},
        )
        xlim = ax[0].get_xlim()
        ax[0].plot([-10,10000], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)

        for i,c in enumerate(order):
            for j,l in enumerate(constants.MODULAR_LAYERS[::-1]):
                if l in c:
                    ax[1].plot([i], [j], marker="o", color='#e64b35' if l==recurrent_label else "#8491b4")

        ax[0].set_xlim(xlim)
        ax[0].set_ylim(48,102)
        ax[1].set_ylim(48,102)
        ax[0].set_xlabel("")
        ax[0].set_ylabel("Performance (% correct)")
        ax[0].set_xticks([])
        ax[1].set_yticks(range(len(constants.MODULAR_LAYERS)), constants.MODULAR_LAYERS[::-1], rotation=20)
        ax[1].set_ylim(-1, len(constants.MODULAR_LAYERS)-.5)
        ax[1].set_aspect('auto')
        ax[1].set_xlabel("Combinations of targets")
        ax[1].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax[1].xaxis.set_ticks_position('none')
        ax[1].yaxis.set_ticks_position('none')
        ax[0].spines[['right', 'top', 'bottom']].set_visible(False)
        ax[0].tick_params(top=False,
               bottom=False,
               left=True,
               right=False,
               labelleft=True,
               labelbottom=True)


        fig.tight_layout()



        return fig, ax



class DebugAnalysis(AbstractAnalysis):
    def analyse(self):
        for n_hebb in [0]:#1,0]:
            for seed in range(50,55):
                plot_df = self.results[(self.results['n_hebb']==n_hebb) & (self.results['seed']==seed)]
                print(plot_df.columns)
                print(n_hebb, seed)
                results = plot_df.iloc[0].test_corrects.mean(axis=1)
                plt.plot(results)
                plt.show()



class Analysis(AbstractAnalysis):
    def analyse(self):

        show = True

        # self.results = self.results[(.00005 < self.results["lr"]) & (self.results["lr"] < .00015)]

        ##############################
        # default learning rate
        ##############################
        for recurrent_label in ["DG", "CA3", "CA1"]:
            df = self.allrecurrent_results
            df = df[(df["n_targets"]==1) & (df["recurrent_label"]==recurrent_label) & (df[recurrent_label]==1)]
            sns.scatterplot(data=df, x="lr", y="mean_test_corrects")
            plt.show()

        ##############################
        # Default all target_labels plot
        ##############################
        for recurrent_label in ["DG", "CA3", "CA1"]:
            fig, ax = self.all_targets_plot(recurrent_label)
            utils.make_fig(fig, ax, self.figures_path, recurrent_label+"all_targets", show=show)

        ##############################
        # default vs no mhn vs manyshots vs manyshots no mhn
        ##############################
        fig, ax = self.mhn_vs_nomhn_plot()
        utils.make_fig(fig, ax, self.figures_path, "default_mhn_vs_nomhn_vs_manyshots_mhn_vs_manyshots_nomhn", show=show)

        ##############################
        # default target_recurrent_delta
        ##############################
        fig, ax = self.target_recurrent_delta_plot()
        utils.make_fig(fig, ax, self.figures_path, "default_target_recurrent_delta", show=show)

        ##############################
        # default target_labels
        ##############################
        df = self.default_results
        fig, ax = self.target_labels_pointplot(df)
        utils.make_fig(fig, ax, self.figures_path, "default_target_labels", show=show)
        fig, ax = self.target_labels_joyplot(df)
        utils.make_fig(fig, ax, self.figures_path, "default_target_labels_joy", show=show)


        ##############################
        # nomhn target_labels
        ##############################
        df = self.nomhn_results
        fig, ax = self.target_labels_pointplot(df)
        utils.make_fig(fig, ax, self.figures_path, "nomhn_target_labels", show=show)
        fig, ax = self.target_labels_joyplot(df)
        utils.make_fig(fig, ax, self.figures_path, "nomhn_target_labels_joy", show=show)


        ##############################
        # all recurrent_labels target_labels
        ##############################
        for recurrent_label in ["DG", "CA3", "CA1"]:
            df = self.allrecurrent_results
            df = df[df["recurrent_label"]==recurrent_label]
            fig, ax = self.target_labels_pointplot(df)
            utils.make_fig(fig, ax, self.figures_path, recurrent_label+"recurrent_target_labels", show=show)
            fig, ax = self.target_labels_joyplot(df)
            utils.make_fig(fig, ax, self.figures_path, recurrent_label+"recurrent_target_labels_joy", show=show)


        ##############################
        # CA1 recurrent no MSP
        ##############################
        df = self.CA1rec_nomsp_results
        fig, ax = self.target_labels_pointplot(df)
        utils.make_fig(fig, ax, self.figures_path, "CA1_recurrent_target_nomsp_labels", show=show)
        fig, ax = self.target_labels_joyplot(df)
        utils.make_fig(fig, ax, self.figures_path, "CA1_recurrent_target_nomsp_labels_joy", show=show)



        ##############################
        # default target_labels
        ##############################
        df = self.default_results
        fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(4,3), sharey=True, gridspec_kw={'width_ratios':[5,1]})
        df = df[((df['n_targets']==1) | (df['target_labels']==('CA1', 'ECout')))]
        df["target_labels"] = df["target_labels"].astype(str)
        order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)"]#, "('CA1', 'ECout')"]
        x='target_labels'
        y='mean_test_corrects'

        sns.boxplot(
            data=df[df['n_targets']<2],
            ax=ax[0],
            x=x,
            y=y,
            showfliers = True,
            flierprops=flierprops,
            order=order,
            boxprops={"facecolor": (.4, .6, .8, .25)})

        print(df[(df['target_labels']==('CA1', 'ECout'))])
        sns.boxplot(
            data=df[(df['target_labels']==('CA1', 'ECout'))],
            ax=ax[1],
            x=x,
            y=y,
            flierprops=flierprops,
            showfliers = True,
            boxprops={"facecolor": (.4, .6, .8, .25)})
        
        # Annotations for statistical significance
        # pairs = list(combinations(order, 2))

        # df = df[df['n_targets']<2]
        # spearmanres = spearmanr([order.index(target) for target in df['target_labels']], df[y], alternative="greater")
        # print("N:", len(df.index), "|", spearmanres)

        # pairs = [("('ECin',)","('DG',)"), ("('ECin',)","('CA3',)"), ("('ECin',)","('CA1',)"), ("('ECin',)","('ECout',)"), ("('CA1',)","('ECout',)"), ("('CA1',)","('CA1', 'ECout')"), ("('ECout',)","('CA1', 'ECout')")]
        # stattest = {}
        # for p in pairs:
        #     alternative = "two-sided" if p==("('CA1',)","('ECout',)") else "less"
        #     print(p, alternative)
        #     stattest[p] = wilcoxon(
        #         df[df[x]==p[0]][y],
        #         df[df[x]==p[1]][y],
        #         alternative=alternative,
        #         method='approx'
        #     )
        # for p,v in stattest.items():
        #     for i in range(2):
        #         print(p[i], "N:", len(df[df[x]==p[i]][y].index), "Median:", df[df[x]==p[i]][y].median())
        #     print("pvalue", v.pvalue)
        #     print("statistic", v.statistic)
        #     print("z statistic", v.zstatistic)
        #     print("\n\n")
        # # annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        # # annotator.configure(text_format="star", loc="outside")
        # # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])


        for i in range(2):
            xlim = ax[i].get_xlim()
            ax[i].plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
            ax[i].set_xlim(xlim)
            ax[i].set_xlabel("")
        ax[1].set_ylabel("")
        # ax[1].set_ylim(ax[0].get_ylim())
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Target(s)")
        ax[0].set_ylabel("Performance (% correct)")
        ax[0].set_xticklabels([t.get_text()[2:-3] for t in ax[0].get_xticklabels()])
        ax[1].set_xticklabels(["CA1 & ECout"])
        ax[0].set_ylim(48,102)
        ax[1].set_ylim(48,102)
        fig.tight_layout()
        plt.show()

        ##############################
        # no mhn target_labels
        ##############################
        df = self.nomhn_results
        fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(4,3), sharey=True, gridspec_kw={'width_ratios':[5,1]})
        df = df[((df['n_targets']==1) | (df['target_labels']==('CA1', 'ECout')))]
        df["target_labels"] = df["target_labels"].astype(str)
        order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)"]#, "('CA1', 'ECout')"]
        x='target_labels'
        y='mean_test_corrects'

        sns.boxplot(
            data=df[df['n_targets']<2],
            ax=ax[0],
            x=x,
            y=y,
            showfliers = True,
            flierprops=flierprops,
            order=order,
            boxprops={"facecolor": (.4, .6, .8, .25)})

        print(df[(df['target_labels']==('CA1', 'ECout'))])
        sns.boxplot(
            data=df[(df['target_labels']==('CA1', 'ECout'))],
            ax=ax[1],
            x=x,
            y=y,
            flierprops=flierprops,
            showfliers = True,
            boxprops={"facecolor": (.4, .6, .8, .25)})
        
        # Annotations for statistical significance
        # pairs = list(combinations(order, 2))

        # df = df[df['n_targets']<2]
        # spearmanres = spearmanr([order.index(target) for target in df['target_labels']], df[y], alternative="greater")
        # print("N:", len(df.index), "|", spearmanres)

        # pairs = [("('ECin',)","('DG',)"), ("('ECin',)","('CA3',)"), ("('ECin',)","('CA1',)"), ("('ECin',)","('ECout',)"), ("('CA1',)","('ECout',)"), ("('CA1',)","('CA1', 'ECout')"), ("('ECout',)","('CA1', 'ECout')")]
        # stattest = {}
        # for p in pairs:
        #     alternative = "two-sided" if p==("('CA1',)","('ECout',)") else "less"
        #     print(p, alternative)
        #     stattest[p] = wilcoxon(
        #         df[df[x]==p[0]][y],
        #         df[df[x]==p[1]][y],
        #         alternative=alternative,
        #         method='approx'
        #     )
        # for p,v in stattest.items():
        #     for i in range(2):
        #         print(p[i], "N:", len(df[df[x]==p[i]][y].index), "Median:", df[df[x]==p[i]][y].median())
        #     print("pvalue", v.pvalue)
        #     print("statistic", v.statistic)
        #     print("z statistic", v.zstatistic)
        #     print("\n\n")
        # # annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        # # annotator.configure(text_format="star", loc="outside")
        # # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])


        for i in range(2):
            xlim = ax[i].get_xlim()
            ax[i].plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
            ax[i].set_xlim(xlim)
            ax[i].set_xlabel("")
        ax[1].set_ylabel("")
        # ax[1].set_ylim(ax[0].get_ylim())
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Target(s)")
        ax[0].set_ylabel("Performance (% correct)")
        ax[0].set_xticklabels([t.get_text()[2:-3] for t in ax[0].get_xticklabels()])
        ax[1].set_xticklabels(["CA1 & ECout"])
        ax[0].set_ylim(48,102)
        ax[1].set_ylim(48,102)
        fig.tight_layout()
        plt.show()



        ##############################
        # default vs no mhn (target is CA3)
        ##############################
        df = pd.concat([self.default_results, self.nomhn_results])
        df = df[df["target_labels"]==('CA3',)]
        print(df["target_labels"])
        ax = sns.violinplot(
            data=df,
            x="use_mhn",
            y="mean_test_corrects")
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()

        ##############################
        # many shots vs many shots no mhn
        ##############################
        df = pd.concat([self.manyshots_results, self.manyshotsnomhn_results])
        ax = sns.violinplot(
            data=df,
            x="use_mhn",
            y="mean_test_corrects")
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()

        ##############################
        # manyshots no mhn target_labels
        ##############################
        df = self.manyshotsnomhn_results
        flierprops = dict(marker='o', markersize=2)
        fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(4,3), sharey=True, gridspec_kw={'width_ratios':[5,1]})
        df = df[((df['n_targets']==1) | (df['target_labels']==('CA1', 'ECout')))]
        df["target_labels"] = df["target_labels"].astype(str)
        order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)"]#, "('CA1', 'ECout')"]
        x='target_labels'
        y='mean_test_corrects'

        sns.boxplot(
            data=df[df['n_targets']<2],
            ax=ax[0],
            x=x,
            y=y,
            showfliers = True,
            flierprops=flierprops,
            order=order,
            boxprops={"facecolor": (.4, .6, .8, .25)})


        sns.boxplot(
            data=df[(df['target_labels']==('CA1', 'ECout'))],
            ax=ax[1],
            x=x,
            y=y,
            flierprops=flierprops,
            showfliers = True,
            boxprops={"facecolor": (.4, .6, .8, .25)})
        
        # Annotations for statistical significance
        # pairs = list(combinations(order, 2))

        # df = df[df['n_targets']<2]
        # spearmanres = spearmanr([order.index(target) for target in df['target_labels']], df[y], alternative="greater")
        # print("N:", len(df.index), "|", spearmanres)

        # pairs = [("('ECin',)","('DG',)"), ("('ECin',)","('CA3',)"), ("('ECin',)","('CA1',)"), ("('ECin',)","('ECout',)"), ("('CA1',)","('ECout',)"), ("('CA1',)","('CA1', 'ECout')"), ("('ECout',)","('CA1', 'ECout')")]
        # stattest = {}
        # for p in pairs:
        #     alternative = "two-sided" if p==("('CA1',)","('ECout',)") else "less"
        #     print(p, alternative)
        #     stattest[p] = wilcoxon(
        #         df[df[x]==p[0]][y],
        #         df[df[x]==p[1]][y],
        #         alternative=alternative,
        #         method='approx'
        #     )
        # for p,v in stattest.items():
        #     for i in range(2):
        #         print(p[i], "N:", len(df[df[x]==p[i]][y].index), "Median:", df[df[x]==p[i]][y].median())
        #     print("pvalue", v.pvalue)
        #     print("statistic", v.statistic)
        #     print("z statistic", v.zstatistic)
        #     print("\n\n")
        # # annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        # # annotator.configure(text_format="star", loc="outside")
        # # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])


        for i in range(2):
            xlim = ax[i].get_xlim()
            ax[i].plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
            ax[i].set_xlim(xlim)
            ax[i].set_xlabel("")
        ax[1].set_ylabel("")
        # ax[1].set_ylim(ax[0].get_ylim())
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Target(s)")
        ax[0].set_ylabel("Performance (% correct)")
        ax[0].set_xticklabels([t.get_text()[2:-3] for t in ax[0].get_xticklabels()])
        ax[1].set_xticklabels(["CA1 & ECout"])
        ax[0].set_ylim(48,102)
        ax[1].set_ylim(48,102)
        fig.tight_layout()
        plt.show()

        ##############################
        # n_targets plot
        ##############################
        df = self.default_results
        flierprops = dict(marker='o', markersize=2)
        df["n_targets"] = df["n_targets"].astype(int).astype(str)
        order = [str(i) for i in range(int(df["n_targets"].max()) + 1)]
        x='n_targets'
        y='mean_test_corrects'
        plt.figure(figsize=(3,3))
        ax = sns.boxplot(
            data=df,
            x=x,
            y=y,
            showfliers=True,
            order=order,
            flierprops=flierprops,
            boxprops={"facecolor": (.4, .6, .8, .25)})
        xlim = ax.get_xlim()
        plt.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
        ax.set_xlim(xlim)
        ax.set_ylim(48,102)

        # Annotations for statistical significance

        spearmanres = spearmanr(df['n_targets'].astype(str).astype(int), df[y], alternative="greater")
        print("N:", len(df.index), "|", spearmanres)

        pairs = [("0","1"), ("1","2"), ("2","5")]
        if len(pairs)>0:
            # print("0 better than chance:", wilcoxon(
            #     plot_df[plot_df[x]=="0"][y].to_numpy()-50,
            #     alternative="greater"
            # ))
            stattest = {
                p:mannwhitneyu(
                    df[df[x]==p[0]][y],
                    df[df[x]==p[1]][y],
                    alternative="less"
                ) for p in pairs}

            for p,v in stattest.items():
                for i in range(2):
                    print(p[i], "N:", len(df[df[x]==p[i]][y].index), "Median:", df[df[x]==p[i]][y].median())
                print(v)
                print("\n\n")
            # annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
            # annotator.configure(text_format="star", loc="inside")
            # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])

        ax.set_xlim(xlim)
        ax.set_xlabel("Number of targets")
        ax.set_ylabel("Performance (% correct)")
        ax.set_ylim(48,102)
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()
        utils.make_fig(fig, ax, self.figures_path, "ntargets")

        


        df = self.manyshots_results
        ax = sns.violinplot(
            data=df,
            x="use_mhn",
            y="mean_test_corrects")
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()


        df = pd.concat([self.default_results, self.nomhn_results])
        ax = sns.violinplot(
            data=df,
            x="use_mhn",
            y="mean_test_corrects")
        fig = ax.get_figure()
        fig.tight_layout()
        plt.show()

        
        

        

        

        
        

        assert False

        for msp in [False, True]:
            for n_hebb in [1,0]:

                # self.results = self.results[self.results["lr"]>10**-4.5]

                # for param in ["seed","N_scale","lr","kappa","lamb","eta"]:
                #     data = self.results[(self.results['n_hebb']==n_hebb) & (self.results['msp']==msp) & (self.results['mechanisms']==6)]
                #     ax = sns.scatterplot(data=data, x=param, y="mean_test_corrects", size=.2, alpha=.5)
                #     if param in ["lr"]:
                #         ax.set(xscale="log")
                #     fig = ax.get_figure()
                #     fig.tight_layout()
                #     utils.make_fig(fig, ax, self.figures_path, param+"_nhebb_"+str(n_hebb)+"_msp_"+str(msp))



                ##############################
                # contribution plot
                ##############################
                # control_perf = {l:[] for l in constants.MODULAR_LAYERS}
                # targeted_perf = {l:[] for l in constants.MODULAR_LAYERS}
                contribution = []
                for i,layer in enumerate(constants.MODULAR_LAYERS):
                    for n_targets in range(1,len(constants.MODULAR_LAYERS)):
                        for _,control in self.results[(self.results['n_hebb']==n_hebb) & (self.results['msp']==msp) & (self.results['mechanisms']==6) & (self.results['n_targets']==n_targets) & (self.results[layer]==0)].iterrows():
                            targeted = self.results[(self.results['n_hebb']==n_hebb) & (self.results['msp']==msp) & (self.results['mechanisms']==6) & (self.results['n_targets']==n_targets+1) & (self.results[layer]==1)]

                            # make sure we focus on the same parameters
                            targeted = targeted[(targeted.seed==control.seed) & (targeted.N_scale==control.N_scale) & (targeted.lr==control.lr) & (targeted.kappa==control.kappa) & (targeted.lamb==control.lamb) & (targeted.eta==control.eta)]

                            # make sure the same layers are also modulated
                            for target in [t for t in constants.MODULAR_LAYERS if control[t]==1]:
                                targeted = targeted[targeted[target]==1]

                            # control_perf[layer].append(control.mean_test_corrects)
                            # targeted_perf[layer].append(targeted.mean_test_corrects)
                            contribution.append({"Target":layer, "Control":float(control.mean_test_corrects), "Targeted":float(targeted.mean_test_corrects), "Contribution":float(targeted.mean_test_corrects)-float(control.mean_test_corrects)})

                #             plt.plot([i*2, i*2+1], [control.mean_test_corrects, targeted.mean_test_corrects], color="gray", alpha=.2)
                # plt.show()
                # plt.close()

                contribution = pd.DataFrame(contribution)
                print(contribution["Contribution"].mean())
                flierprops = dict(marker='o', markersize=2)
                plt.figure(figsize=(3,3))
                for layer in constants.MODULAR_LAYERS:
                    print(layer, contribution[contribution["Target"]==layer]["Contribution"].mean())
                ax = sns.violinplot(
                    data=contribution,
                    x="Target",
                    y="Contribution",
                    showfliers=False,
                    order=constants.MODULAR_LAYERS,
                    flierprops=flierprops,
                    boxprops={"facecolor": (.4, .6, .8, .25)})
                fig = ax.get_figure()
                fig.tight_layout()
                utils.make_fig(fig, ax, self.figures_path, "contribution"+"_nhebb_"+str(n_hebb)+"_msp_"+str(msp))


                fig = plt.figure(figsize=(9,3))
                for i,layer in enumerate(constants.MODULAR_LAYERS):
                    layer_contribution = contribution[contribution["Target"]==layer]
                    ys = np.array([list(layer_contribution["Control"]),list(layer_contribution["Targeted"])])
                    plt.plot([[i*2 for _ in range(ys.shape[1])], [i*2+1 for _ in range(ys.shape[1])]], ys, color=(.4, .6, .8), alpha=.25)
                    sns.boxplot(
                        x=[i*2]*len(list(layer_contribution["Control"])),
                        y=layer_contribution["Control"],
                        ax=plt.gca(),
                        flierprops=flierprops,
                        boxprops={"facecolor": (.4, .6, .8, .25)},
                        width=.2,
                        order=range(len(constants.MODULAR_LAYERS)*2))
                    sns.boxplot(
                        x=[i*2+1]*len(list(layer_contribution["Targeted"])),
                        y=layer_contribution["Targeted"],
                        ax=plt.gca(),
                        flierprops=flierprops,
                        boxprops={"facecolor": (.4, .6, .8, .25)},
                        width=.2,
                        order=range(len(constants.MODULAR_LAYERS)*2))
                fig.tight_layout()
                utils.make_fig(fig, fig.axes, self.figures_path, "control_vs_targeted"+"_nhebb_"+str(n_hebb)+"_msp_"+str(msp))


                ##############################
                # n_targets plot
                ##############################

                plot_df = self.results[(self.results['n_hebb']==n_hebb) & (self.results['msp']==msp) & ((self.results['mechanisms']==6) | (self.results['mechanisms']==-1))]
                plot_df["n_targets"] = plot_df["n_targets"].astype(int).astype(str)
                order = [str(i) for i in range(int(plot_df["n_targets"].max()) + 1)]
                x='n_targets'
                y='mean_test_corrects'
                plt.figure(figsize=(3,3))
                ax = sns.boxplot(
                    data=plot_df,
                    x=x,
                    y=y,
                    showfliers=True,
                    order=order,
                    flierprops=flierprops,
                    boxprops={"facecolor": (.4, .6, .8, .25)})
                xlim = ax.get_xlim()
                plt.plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
                ax.set_xlim(xlim)
                ax.set_ylim(48,102)

                # Annotations for statistical significance

                spearman_df = plot_df
                spearmanres = spearmanr(spearman_df['n_targets'].astype(str).astype(int), spearman_df[y], alternative="greater")
                print("N:", len(spearman_df.index), "|", spearmanres)

                pairs = [("1","2"), ("2","5")]#[("0","1")]
                if len(pairs)>0:
                    # print("0 better than chance:", wilcoxon(
                    #     plot_df[plot_df[x]=="0"][y].to_numpy()-50,
                    #     alternative="greater"
                    # ))
                    stattest = {
                        p:mannwhitneyu(
                            plot_df[plot_df[x]==p[0]][y],
                            plot_df[plot_df[x]==p[1]][y],
                            alternative="less"
                        ) for p in pairs}

                    for p,v in stattest.items():
                        for i in range(2):
                            print(p[i], "N:", len(plot_df[plot_df[x]==p[i]][y].index), "Median:", plot_df[plot_df[x]==p[i]][y].median())
                        print(v)
                        print("\n\n")
                    # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
                    # annotator.configure(text_format="star", loc="inside")
                    # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])

                ax.set_xlim(xlim)
                ax.set_xlabel("Number of targets")
                ax.set_ylabel("Performance (% correct)")
                ax.set_ylim(48,102)
                fig = ax.get_figure()
                fig.tight_layout()
                utils.make_fig(fig, ax, self.figures_path, "ntargets"+"_nhebb_"+str(n_hebb)+"_msp_"+str(msp))

                # # fig, ax = plt.subplots()
                # plot_df = df[(df['n_hebb']==n_hebb) & (df['mechanisms']==6) & (df['n_targets']>0)]
                # ax = sns.regplot(data=plot_df, x=x, y=y, x_estimator=np.mean)
                #
                # # corr_dfs = [df[(df['n_hebb']==n_hebb) & (df['mechanisms']==6) & (df['n_targets']>min) & (df['n_targets']<max)] for min,max in [(0,4), (2,6)]]
                # # print(spearmanr(corr_dfs[0][x], corr_dfs[0][y], alternative='greater'))
                # # print(spearmanr(corr_dfs[1][x], corr_dfs[1][y], alternative='less'))
                # print(spearmanr(plot_df[x], plot_df[y], alternative='greater'))
                #
                # reg = LinearRegression().fit(plot_df[x].to_numpy()[:,None], plot_df[y])
                # print(reg.score(plot_df[x].to_numpy()[:,None], plot_df[y]))
                #
                # mod = sm.OLS(plot_df[y], plot_df[x].to_numpy()[:,None])
                # fii = mod.fit()
                # print(fii.summary2())
                # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # ax.set_xlabel("Number of modulated layers")
                # ax.set_ylabel("Performance (% correct)")
                # fig = ax.get_figure()

                # # Annotations for statistical significance
                # plot_df = df[(df['n_hebb']==n_hebb) & ((df['mechanisms']==6) | (df['mechanisms']==-1))]
                # plot_df["n_targets"] = plot_df["n_targets"].astype(int).astype(str)
                # pairs = [("1","2"),("2","3"),("4","3"),("5","4"),("5","3")]
                # p_values = {
                #     p:mannwhitneyu(
                #         plot_df[plot_df[x]==p[0]][y],
                #         plot_df[plot_df[x]==p[1]][y],
                #         alternative="less"
                #     ).pvalue for p in pairs}
                # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=[str(i) for i in range(int(plot_df["n_targets"].max()) + 1)])
                # annotator.configure(text_format="star", loc="inside")
                # annotator.set_pvalues_and_annotate(list(p_values.values()))
                # utils.make_fig(fig, ax, "fig/seaborn")

                ##############################
                # target_labels plot
                ##############################


                fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(4,3), sharey=True, gridspec_kw={'width_ratios':[5,1]})
                plot_df = self.results[((self.results['n_targets']<2) | (self.results['target_labels']==('CA1', 'ECout'))) & (self.results['n_hebb']==n_hebb) & (self.results['msp']==msp) & (self.results['mechanisms']==6)]
                plot_df["target_labels"] = plot_df["target_labels"].astype(str)
                order = ["('ECin',)", "('DG',)", "('CA3',)", "('CA1',)", "('ECout',)"]#, "('CA1', 'ECout')"]
                x='target_labels'
                y='mean_test_corrects'

                sns.boxplot(
                    data=plot_df[plot_df['n_targets']<2],
                    ax=ax[0],
                    x=x,
                    y=y,
                    showfliers = True,
                    flierprops=flierprops,
                    order=order,
                    boxprops={"facecolor": (.4, .6, .8, .25)})


                sns.boxplot(
                    data=plot_df[(self.results['target_labels']==('CA1', 'ECout'))],
                    ax=ax[1],
                    x=x,
                    y=y,
                    flierprops=flierprops,
                    showfliers = True,
                    boxprops={"facecolor": (.4, .6, .8, .25)})

                # Annotations for statistical significance
                # pairs = list(combinations(order, 2))

                spearman_df = plot_df[plot_df['n_targets']<2]
                spearmanres = spearmanr([order.index(target) for target in spearman_df['target_labels']], spearman_df[y], alternative="greater")
                print("N:", len(spearman_df.index), "|", spearmanres)

                pairs = [("('ECin',)","('DG',)"), ("('ECin',)","('CA3',)"), ("('ECin',)","('CA1',)"), ("('ECin',)","('ECout',)"), ("('CA1',)","('ECout',)"), ("('CA1',)","('CA1', 'ECout')"), ("('ECout',)","('CA1', 'ECout')")]
                stattest = {}
                for p in pairs:
                    alternative = "two-sided" if p==("('CA1',)","('ECout',)") else "less"
                    print(p, alternative)
                    stattest[p] = wilcoxon(
                        plot_df[plot_df[x]==p[0]][y],
                        plot_df[plot_df[x]==p[1]][y],
                        alternative=alternative,
                        method='approx'
                    )
                for p,v in stattest.items():
                    for i in range(2):
                        print(p[i], "N:", len(plot_df[plot_df[x]==p[i]][y].index), "Median:", plot_df[plot_df[x]==p[i]][y].median())
                    print("pvalue", v.pvalue)
                    print("statistic", v.statistic)
                    print("z statistic", v.zstatistic)
                    print("\n\n")
                # # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
                # # annotator.configure(text_format="star", loc="outside")
                # # annotator.set_pvalues_and_annotate([v.pvalue for v in stattest.values()])


                for i in range(2):
                    xlim = ax[i].get_xlim()
                    ax[i].plot([-10,10], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)
                    ax[i].set_xlim(xlim)
                    ax[i].set_xlabel("")
                ax[1].set_ylabel("")
                # ax[1].set_ylim(ax[0].get_ylim())
                # add a big axis, hide frame
                fig.add_subplot(111, frameon=False)
                # hide tick and tick label of the big axis
                plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
                plt.xlabel("Target(s)")
                ax[0].set_ylabel("Performance (% correct)")
                ax[0].set_xticklabels([t.get_text()[2:-3] for t in ax[0].get_xticklabels()])
                ax[1].set_xticklabels(["CA1 & ECout"])
                ax[0].set_ylim(48,102)
                ax[1].set_ylim(48,102)
                fig.tight_layout()
                utils.make_fig(fig, ax, self.figures_path, "target_labels"+"_nhebb_"+str(n_hebb)+"_msp_"+str(msp))

                ##############################
                # All target_labels plot
                ##############################

                plot_df = self.results[(self.results['n_hebb']==n_hebb) & (self.results['msp']==msp) & (self.results['mechanisms']==6) &
                    (
                        (self.results['n_targets']>0)
                    )
                ]
                plot_df["target_labels"] = plot_df["target_labels"].astype(str)
                order = list(plot_df.groupby(by=["target_labels"])["mean_test_corrects"].quantile(.25).sort_values(ascending=False).index)
                x='target_labels'
                y='mean_test_corrects'

                fig,ax = plt.subplots(nrows=2, ncols=1, figsize=(5,3.5), sharex=True, gridspec_kw={'height_ratios':[3,2]})

                sns.boxplot(
                    data=plot_df,
                    x=x,
                    y=y,
                    ax=ax[0],
                    showfliers = True,
                    flierprops=flierprops,
                    order=order,
                    boxprops={"facecolor": (.4, .6, .8, .25)})
                xlim = ax[0].get_xlim()
                ax[0].plot([-10,10000], [50 for i in range(2)], linestyle='--', color='k', alpha=.2, zorder=-1)

                for i,c in enumerate(order):
                    for j,l in enumerate(constants.MODULAR_LAYERS[::-1]):
                        if l in c:
                            ax[1].plot([i], [j], marker="x", color='black')


                # # Annotations for statistical significance
                # # pairs = list(combinations(order, 2))
                # p_values = {
                #     p:mannwhitneyu(
                #         plot_df[plot_df[x]==p[0]][y],
                #         plot_df[plot_df[x]==p[1]][y],
                #         alternative="less"
                #     ).pvalue for p in pairs}
                # print("pairs",pairs)
                # annotator = Annotator(ax, pairs, data=plot_df, x=x, y=y, order=order)
                # annotator.configure(text_format="star", loc="inside")
                # annotator.set_pvalues_and_annotate(list(p_values.values()))

                ax[0].set_xlim(xlim)
                ax[0].set_ylim(48,102)
                ax[1].set_ylim(48,102)
                ax[0].set_xlabel("")
                ax[0].set_ylabel("Performance (% correct)")
                ax[0].set_xticks([])
                ax[1].set_yticks(range(len(constants.MODULAR_LAYERS)), constants.MODULAR_LAYERS[::-1], rotation=20)
                ax[1].set_ylim(-1, len(constants.MODULAR_LAYERS)-.5)
                ax[1].set_aspect('auto')
                ax[1].set_xlabel("Combinations of targets")
                ax[1].spines.right.set_visible(False)
                ax[1].spines.left.set_visible(False)
                ax[1].spines.top.set_visible(False)
                ax[1].spines.bottom.set_visible(False)
                ax[1].xaxis.set_ticks_position('none')
                ax[1].yaxis.set_ticks_position('none')
                fig.tight_layout()
                plt.subplots_adjust(hspace=.01)
                utils.make_fig(fig, ax, self.figures_path, "alltargets"+"_nhebb_"+str(n_hebb)+"_msp_"+str(msp))



def analyse_splitters(xp_df, start_session=50, end_session=None, n_contexts=2, n_tasks=8, target_labels_list=None):

    organized_activity = {target_labels: {} for target_labels in xp_df['target_labels'].unique()}
    splitters = {target_labels: {} for target_labels in xp_df['target_labels'].unique()}

    if end_session is None:
        end_session = xp_df.iloc[0]['n_train_sessions']

    sessions = np.arange(start_session, end_session, dtype=int)
    tasks = np.arange(n_tasks, dtype=int)
    contexts = np.arange(n_contexts, dtype=int)

    proportion_dicts = []

    for i,target_labels in enumerate(target_labels_list):


        tmp_df = xp_df[(xp_df['target_labels']==target_labels)]
        assert len(tmp_df.index)==1

        proportion_dict = {}

        for layer in constants.MODULAR_LAYERS:

            activity = tmp_df.iloc[0]['test_activity'][layer]

            n_neurons = activity.shape[2]


            organized_activity[target_labels][layer] = organize_activity(
                activity=activity,
                tasks=tasks,
                contexts=contexts,
                sessions=sessions,
                n_neurons=n_neurons
            )

            splitters[target_labels][layer] = find_splitters(
                activity_df=organized_activity[target_labels][layer],
                tasks=tasks,
                contexts=contexts,
                n_neurons=n_neurons
            )


    org_act_without_splitters, splitter_count = remove_splitters(organized_activity, splitters, n_tasks)
    org_act_without_random = remove_random(organized_activity, splitter_count)

    for plot_func in [pca_hist, pca_scatter]:
        splitter_pca(
            organized_activity=organized_activity,
            target_labels_list=target_labels_list,
            tasks=tasks,
            contexts=contexts,
            plot_func=plot_func,
            name="pca",
            in_3d=False
        )

        splitter_pca(
            organized_activity=org_act_without_splitters,
            target_labels_list=target_labels_list,
            tasks=tasks,
            contexts=contexts,
            plot_func=plot_func,
            name="pca_without_splitters",
            in_3d=False
        )

        splitter_pca(
            organized_activity=org_act_without_random,
            target_labels_list=target_labels_list,
            tasks=tasks,
            contexts=contexts,
            plot_func=plot_func,
            name="pca_without_random",
            in_3d=False
        )

    proportion_plot(fig, ax, splitters)

    n_tasks_plot(
        splitter_df=pd.concat(splitter_dfs),
        tasks=tasks
    )

def remove_splitters(organized_activity, splitters, n_tasks):

    org_act_without_splitters = {target_labels: {} for target_labels in organized_activity.keys()}
    splitter_count = {target_labels: {} for target_labels in organized_activity.keys()}

    for target_labels_k, target_labels_v in organized_activity.items():
        for layer_k, layer_v in target_labels_v.items():
            splitters_df = splitters[target_labels_k][layer_k]
            splitters_list = list(splitters_df[splitters_df["Total"]==n_tasks]["#"])

            org_act_without_splitters[target_labels_k][layer_k] = layer_v.drop(['#'+str(i) for i in splitters_list], axis=1)
            splitter_count[target_labels_k][layer_k] = len(splitters_list)

    return org_act_without_splitters, splitter_count

def remove_random(organized_activity, splitter_count, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    org_act_without_random = {target_labels: {} for target_labels in organized_activity.keys()}
    for target_labels_k, target_labels_v in organized_activity.items():
        for layer_k, layer_v in target_labels_v.items():

            n_neurons = int(layer_v.columns[-1][1:]) + 1
            to_remove = rng.choice(n_neurons, size=splitter_count[target_labels_k][layer_k], replace=False)
            org_act_without_random[target_labels_k][layer_k] = layer_v.drop(['#'+str(i) for i in to_remove], axis=1)

    return org_act_without_random

def organize_activity(activity, tasks, contexts, sessions, n_neurons):

    organized_activity = []

    for task in tasks:
        for context in contexts:

            task_x_context = task + len(tasks) * context

            for session in sessions:

                # Add information about session, task and context
                organized_activity.append({
                    'session': session,
                    'task': task,
                    'context': context,
                })

                # Add activity of all neurons
                for neuron in range(n_neurons):

                    organized_activity[-1]['#'+str(neuron)] = activity[session, task_x_context, neuron]

    return pd.DataFrame(organized_activity)

def find_splitters(activity_df, tasks, contexts, n_neurons, pvalue_threshold=.001):

    # Stores whether a cell is splitter in a task (1) or not (0)
    splitter = [{t:0 for t in tasks} for n in range(n_neurons)]

    splitter_proportion = {}
    for task in tasks:

        # group by context
        context_dfs = [activity_df[(activity_df['task']==task) & (activity_df['context']==context)] for context in contexts]

        for neuron in range(n_neurons):

            # check if any difference between contexts
            if ((context_dfs[0]['#'+str(neuron)].to_numpy() - context_dfs[1]['#'+str(neuron)].to_numpy())!=0).any():
                test = wilcoxon(
                    context_dfs[0]['#'+str(neuron)],
                    context_dfs[1]['#'+str(neuron)],
                    alternative="two-sided")

                if test.pvalue < pvalue_threshold:
                    splitter[neuron][task] = 1

    splitter = pd.DataFrame(splitter)
    splitter['Total'] = splitter.sum(axis=1)
    splitter['#'] = range(n_neurons)
    return splitter

def splitter_pca(organized_activity, target_labels_list, tasks, contexts, plot_func, name='pca', in_3d=False):

    subplot_kw = dict(projection='3d') if in_3d else None
    fig, ax = plt.subplots(len(target_labels_list), len(constants.MODULAR_LAYERS), sharex=True, figsize=(10,5),subplot_kw=subplot_kw)

    for i,target_labels in enumerate(target_labels_list):
        for j,layer in enumerate(constants.MODULAR_LAYERS):


            activity_df = organized_activity[target_labels][layer]
            training_data = activity_df[[col for col in activity_df.columns if '#' in col]]

            #Scale the data
            scaler = StandardScaler()
            scaler.fit(training_data)
            training_data = scaler.transform(training_data)

            #Obtain principal components
            pca = PCA().fit(training_data)

            plot_func(fig, ax[i,j], pca, activity_df, contexts, name, in_3d)
    plt.tight_layout()
    # utils.make_fig(fig, ax, "fig/seaborn/splitter", name)

def pca_hist(fig, ax, pca, activity_df, contexts, name, in_3d):
    pcs = pca.components_
    print(pcs.shape)

    ytrain = activity_df['context']
    Xtrain = activity_df[[col for col in activity_df.columns if '#' in col]]
    print(ytrain.shape, Xtrain.shape)
    Xtrain = pca.transform(Xtrain)#[:,:4]

    print(ytrain.shape, Xtrain.shape)

    for i in range(5):
        spearmanres = spearmanr(Xtrain[:,i], ytrain)
        print(spearmanres)

    ax.hist([pcs[i,:] for i in range(3)])


def pca_scatter(fig, ax, pca, activity_df, contexts, name, in_3d):
    for h,context in enumerate(contexts):
        context_data = activity_df[activity_df['context']==h]
        context_data = context_data[[col for col in activity_df.columns if '#' in col]]
        proj = pca.transform(context_data)

        if in_3d:
            ax.scatter(proj[:,0], proj[:,1], proj[:,2], s=2, c=["red","blue"][h], label=str(context), alpha=.7)
        else:
            ax.scatter(proj[:,0], proj[:,1], s=2, c=["red","blue"][h], label=str(context), alpha=.7)




def proportion_plot(fig, ax, proportion_dicts):
    fig, ax = plt.subplots(len(organized_activity), 1, sharex=True, figsize=(10,5))
    for i in range(len(proportion_dicts)):
        sns.barplot(data=proportion_dicts[i], x="Layer", y="Proportion", ax=ax[i])

    plt.tight_layout()
    utils.make_fig(fig, ax, "fig/seaborn/splitter", "proportion")

def n_tasks_plot(splitter_df, tasks):
    fig, ax = plt.subplots()
    splitter_df['Total'].hist(grid=False, bins=range(len(tasks)+2), ax=ax)
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    utils.make_fig(fig, ax, "fig/seaborn/splitter", "in_n_tasks")
