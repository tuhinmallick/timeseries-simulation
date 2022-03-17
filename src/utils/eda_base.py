import pandas as pd
import numpy as np
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot, quarter_plot
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

class exploratory_data_analysis():
    
    def __init__(self, target_name: str, df: pd.DataFrame):
        self.target_name = target_name
        self.df = df
        if type(self.df.index ) is pd.DatetimeIndex:
            self.x_date = self.df.index
        else:
            raise ValueError("DataFrame index must be pandas datetime.")
        
        self.y_target = self.df[self.target_name]
        
    def single_timeseries_plot(self, y_variable: str, rolling_mean=False, rolling_std=False, save_path=None, title="", figsize=(14,7), dpi=100, streamlit = False, **kwargs):
        """Function to create a single series timeseries plot for a target variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            rolling_mean (boolean, optional):
            rolling_std
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            title (str, optional): Title of the plot. Defaults to "".
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (14,7).
            dpi (int, optional): DPI value of the plot. Defaults to 100.
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 16
        fontsize_legend = kwargs['fontsize_legend'] if kwargs.get('fontsize_legend') else 14
        rolling_window = kwargs["rolling_window"] if kwargs.get("rolling_window") else 6
        xlabel = kwargs["xlabel"] if kwargs.get("xlabel") else "Date"
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        x_range = kwargs["x_range"] if kwargs.get("x_range") else None
        y_range = kwargs["y_range"] if kwargs.get("y_range") else None
        
        # Generate the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(self.x_date, self.df[y_variable], label=f"Trend {y_variable}")
        if rolling_mean == True:
            ax.plot(self.df[y_variable].rolling(rolling_window).mean(), label=f"Moving Average")

        if rolling_std == True:
            ax.plot(self.df[y_variable].rolling(rolling_window).std(), label=f"Moving Standard Deviation")
        
        if (rolling_mean == True) or (rolling_std == True):
            plt.legend(fontsize=fontsize_legend)
        
        # Set the x and y range:
        if type(x_range) == list:
            ax.set_xlim(datetime.strptime(x_range[0], "%d/%m/%Y").date(), datetime.strptime(x_range[1], "%d/%m/%Y").date())
            
        if type(y_range) == list:
            ax.set_ylim(y_range[0], y_range[1])

        # Plot aesthetics
        ax.set_title(label=title, fontsize=fontsize_title)
        ax.set_xlabel(xlabel=xlabel, fontsize=fontsize_label)
        ax.set_ylabel(ylabel=f"{y_variable.title()}", fontsize=fontsize_label)
        fig.tight_layout()
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"single_timeseries_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def monthly_plot(self, y_variable: str, save_path=None, figsize=(20,7), dpi=80, streamlit = False, **kwargs):
        """Function to plot the monthly trend of a target variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,7).
            dpi (int, optional): DPI value of the plot. Defaults to 80.
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 14
        line_color = kwargs['line_color'] if kwargs.get('line_color') else "cyan"
        zorder = kwargs['zorder'] if kwargs.get('zorder') else 0
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Generate the plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig = month_plot(x=self.df[y_variable].dropna(), ax=ax)
        
        # Plot aesthetics
        ax.set_title(label=f"Month Plot {y_variable.title()}", fontsize=fontsize_title)
        ax.set_xlabel(xlabel="Month", fontsize=fontsize_label)
        ax.set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)
        
        # Change line color
        lines = ax.get_lines()
        for line in lines: line.set_color(line_color)
        for line in lines: line.set_zorder(zorder)   
        fig.tight_layout()
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"monthly_plot_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def quarterly_plot(self, y_variable: str, save_path=None, figsize=(20,7), dpi=80, streamlit = False, **kwargs):
        """Function to plot the quarterly trend of a target variable.

        Args:
        
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,7).
            dpi (int, optional): DPI value of the plot. Defaults to 80.
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 14
        line_color = kwargs['line_color'] if kwargs.get('line_color') else "cyan"
        zorder = kwargs['zorder'] if kwargs.get('zorder') else 0
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Generate the plot
        df_sub = self.df[y_variable].copy()
        df_sub.index = self.df.index.to_period('Q')
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig = quarter_plot(x=df_sub.dropna(), ax=ax)
        
        # Plot aesthetics
        ax.set_title(label=f"Month Plot {y_variable.title()}", fontsize=fontsize_title)
        ax.set_xlabel(xlabel="Month", fontsize=fontsize_label)
        ax.set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)
        
        # Change line color
        lines = ax.get_lines()
        for line in lines: line.set_color(line_color)
        for line in lines: line.set_zorder(zorder)   
        fig.tight_layout()
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"quarterly_plot_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def seasonal_boxplot_ym(self, y_variable: str, save_path=None, figsize=(20,7), dpi=80, streamlit = False, **kwargs):
        """Function that creates the seasonal boxplot for year and month.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,7).
            dpi (int, optional): DPI value of the plot. Defaults to 80.
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 14
        fontsize_ticks = kwargs['fontsize_ticks'] if kwargs.get('fontsize_ticks') else 14
        x_labelrotation = kwargs["x_labelrotation"] if kwargs.get("x_labelrotation") else 45
        box_line_color = kwargs["x_labelrotation"] if kwargs.get("x_labelrotation") else "silver"
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Prepare data for plot by adding year and month column.
        self.df['year'] = [d.year for d in self.df.index]
        self.df['month'] = [d.strftime('%b') for d in self.df.index]

        # Create plots
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
        
        sns.boxplot(
            x='year', 
            y=y_variable, 
            data=self.df, 
            ax=axs[0],
            boxprops=dict(edgecolor=box_line_color),
            capprops=dict(color=box_line_color),
            whiskerprops=dict(color=box_line_color),
            flierprops=dict(color=box_line_color, markerfacecolor=box_line_color, markeredgecolor=box_line_color),
            medianprops=dict(color=box_line_color)
            )
        sns.boxplot(
            x='month', 
            y=y_variable, 
            data=self.df.loc[~self.df.year.isin([1991, 2000]), :],
            ax=axs[1],
            boxprops=dict(edgecolor=box_line_color),
            capprops=dict(color=box_line_color),
            whiskerprops=dict(color=box_line_color),
            flierprops=dict(color=box_line_color, markerfacecolor=box_line_color, markeredgecolor=box_line_color),
            medianprops=dict(color=box_line_color)
            )
        
        # Plot Aesthetics
        axs[0].set_title(label='Year-wise Box Plot\n(The Trend)', fontsize=fontsize_title); 
        axs[1].set_title(label='Month-wise Box Plot\n(The Seasonality)', fontsize=fontsize_title)
        
        axs[0].set_xlabel(xlabel="Year".title(), fontsize=fontsize_label)
        axs[1].set_xlabel(xlabel="Month".title(), fontsize=fontsize_label)
        
        axs[0].set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)
        axs[1].set_ylabel(ylabel=y_variable.title(), fontsize=fontsize_label)
        
        axs[0].tick_params(axis='x', labelsize=fontsize_ticks, labelrotation=x_labelrotation)
        axs[1].tick_params(axis='x', labelsize=fontsize_ticks, labelrotation=x_labelrotation)
        
        axs[0].tick_params(axis='y', labelsize=fontsize_ticks)
        axs[1].tick_params(axis='y', labelsize=fontsize_ticks)
        fig.tight_layout()
        
        # Remove the two helper columns
        self.df.drop(["month", "year"], axis=1, inplace=True)
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"ym_seasonal_decompose_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
            
        plt.show()
    
    def target_lag_plots(self, y_variable: str, lags=8, save_path=None, figsize=(16,7), streamlit = False, **kwargs):
        """Function to create a series of lag plots (number specified by lags) for the specified variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            lags (int, optional): Number of lags to be added. Please not that depending on the number of lags you need to specify the plot_matrix_shape e.g. 240. Defaults to 8.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (16,7).
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        plot_matrix_shape = kwargs['plot_matrix_shape'] if kwargs.get('plot_matrix_shape') else 240
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        plt.figure(figsize=figsize)
        plt.suptitle(f'Lag Correlation Plot for {y_variable}', fontsize=fontsize_title)
        
        # Abstract values and convert to columns for the target varaiable
        values = self.df[y_variable]
        columns = [values]
        
        # Append the lags t+1.
        for i in range(1,(lags + 1)):
            columns.append(values.shift(i))
        df_lag = pd.concat(columns, axis=1)
        columns = ['t+1']

        # Append the lags t-h
        for i in range(1,(lags + 1)):
            columns.append(f't-{i}')
        df_lag.columns = columns
        
        plt.figure(1)
        for i in range(1,(lags + 1)):
            ax = plt.subplot(plot_matrix_shape + i) # this is for dimensions rows, cols
            ax.set_title(f't+1 vs t-{i}')
            plt.scatter(x=df_lag['t+1'].values, y=df_lag[f't-{i}'].values, s=1)
        plt.tight_layout()
        
        if save_path!=None:
            plt.savefig(os.path.join(save_path, f"lag_plot_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()

    def plot_acf_pacf(self, y_variable: str, diff_target=False, lags=60, save_path=None, streamlit = False, figsize=(15,6), **kwargs):
        """Function to create the autocorrelation and partial autocorrelation plot.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            diff_target (bool, optional): Select if the target column is to find the discrete differences. Defaults to False.
            lags (int, optional): Number of lags for the correlation. Defaults to 60.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (15,6).
        """
        # Parse some kwargs configurations
        k_diff = kwargs['k_diff'] if kwargs.get('k_diff') else 1
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Generate plot
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        
        # Check if target variable is differentiated
        if diff_target == True:
            y_target = diff(self.df[y_variable], k_diff=k_diff)
        else:
            y_target = self.df[y_variable].copy()
        
        # Handle missing values by dropping them.
        y_target.dropna(inplace=True)
        
        # Plot acf and pacf
        plot_acf(y_target.tolist(), lags=lags, ax=ax[0]); # just the plot
        plot_pacf(y_target.tolist(), lags=lags, ax=ax[1]); # just the plot
        fig.tight_layout()
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"acf_pacf_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def plot_seasonal_decomposition(self, y_variable: str, save_path=None, figsize=(16,12), streamlit = False, **kwargs):
        """Function to create the seasonal composition plot.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (16,12).
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['k_diff'] if kwargs.get('k_diff') else 20
        extrapolate_trend = kwargs['extrapolate_trend'] if kwargs.get('extrapolate_trend') else "freq"
        decompose_model = kwargs['decompose_model'] if kwargs.get('decompose_model') else "additive"    # Can be "additive", "multiplicative", 
        title_label = kwargs['title_label'] if kwargs.get('title_label') else f"{decompose_model.title()} Decomposition of {y_variable}"
        axhline_color = kwargs['axhline_color'] if kwargs.get('axhline_color') else "white"
        axhline_linewidth = kwargs['axhline_linewidth'] if kwargs.get('axhline_linewidth') else 1.5
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Set plot Params
        plt.rcParams.update(
            {
                'figure.figsize': figsize,
                'lines.markersize': 2,
                }
            )
        
        # Generate plot and plot features
        result_add = seasonal_decompose(self.df[y_variable], model=decompose_model, extrapolate_trend=extrapolate_trend)
        result_add.plot().suptitle(title_label, fontsize=fontsize_title)
        plt.axhline(0, c=axhline_color, linewidth=axhline_linewidth)
        plt.tight_layout()
        
        if save_path!=None:
            plt.savefig(os.path.join(save_path, f"seasonal_decomposition_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def ask_adfuller(self, y_variable: str, autolag="aic", streamlit = False, **kwargs):
        """Function to run ad fuller test on target variable.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            autolag (str, optional): Method to use when automatically determining the lag length among the values. Defaults to "aic". Can be AIC, BIC, t-stat.
        """
        # Parse some kwargs configuration
        regression = kwargs['maxlag'] if kwargs.get('maxlag') else "c" # "c" default, "ct" constant and trend, "ctt" constant linear and quatratic, "n" non constant
        # Run the test:
        test_results = adfuller(self.df[y_variable], regression=regression, autolag=autolag)
        
        print("---------------------------------------------------------------------------------------------------------------------")
        print("AD Fuller Test:")
        print("---------------------------------------------------------------------------------------------------------------------")
        print('Test statistic: ', test_results[0])
        print('p-value: ', test_results[1])
        print('Critical Values:', test_results[4])
        print("----------------------------------------------------------------------------------------------------------------------")
    
        
    def plot_stl_decomposition(self, y_variable: str, seasonal=11, trend=15, save_path=None, streamlit = False, figsize=(16,12), **kwargs):
        """Function to create the seasonal composition plot.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (16,12).
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['k_diff'] if kwargs.get('k_diff') else 20
        title_label = kwargs['title_label'] if kwargs.get('title_label') else f"STL Decomposition of {y_variable}"
        axhline_color = kwargs['axhline_color'] if kwargs.get('axhline_color') else "white"
        axhline_linewidth = kwargs['axhline_linewidth'] if kwargs.get('axhline_linewidth') else 1.5
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Set plot Params
        plt.rcParams.update(
            {
                'figure.figsize': figsize,
                'lines.markersize': 2,
                }
            )
        
        # Generate plot and plot features
        stl = STL(self.df[y_variable], seasonal=seasonal,trend=trend)
        result_add = stl.fit()
        fig = result_add.plot().suptitle(title_label, fontsize=fontsize_title)
        
        # Adjust color of axhline
        fig.axhline(0, c=axhline_color, linewidth=axhline_linewidth)
        fig.tight_layout()
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"stl_decomposition_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def correlate_all_plot(self, y_variable: str, x_variables: list, max_lags=30, streamlit = False, save_path=None, figsize=(20,35), rect=(0,0,1,0.96), **kwargs):
        """Function to create a correlation plot between a target variable y and all the feature variables x. 

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            x_variables (list): Column names of the feature variables in the dataframe. This should exclude y_variable.
            max_lags (int, optional): The maximum number of lags that are used for the correlation plot. Defaults to 30.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (20,35).
            rect (tuple, optional): Tuple that indicates how the tight layout is configured. Defaults to (0,0,1,0.96).
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_sub_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 16
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 14
        n_x_ticks = kwargs["n_x_ticks"] if kwargs.get("n_x_ticks") else 10
        threshold_value = kwargs["threshold_value"] if kwargs.get("threshold_value") else 0.1
        color_fillbetween = kwargs["color_fillbetween"] if kwargs.get("color_fillbetween") else "pink"
        alpha_fillbetween = kwargs["alpha_fillbetween"] if kwargs.get("alpha_fillbetween") else 0.2
        xcorr_lw  = kwargs["xcorr_lw"] if kwargs.get("xcorr_lw") else 2
        usevlines  = kwargs["usevlines"] if kwargs.get("usevlines") else True
        normed  = kwargs["normed"] if kwargs.get("normed") else True
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Generate Plots
        fig, axs = plt.subplots(nrows=int(np.ceil(len(x_variables)/4)), ncols=4, sharex=True, sharey=True, figsize=figsize)
        
        # Generate threshold dataset
        x_threshold = np.arange(-max_lags, max_lags+10, n_x_ticks)
        y_upper_max = [threshold_value]*len(x_threshold)
        y_lower_max = [-threshold_value]*len(x_threshold)
        
        # Reshape if ndim == 1
        if axs.ndim == 1:
            axs = axs.reshape(1, -1)
            
        # Iterate through the the correlation plots. Adding 4 plots per row.
        for i, x_variable in enumerate(x_variables):
            # Add x an y value from dataframe.
            x, y = self.df[x_variable].fillna(0), self.df[y_variable].dropna()
            axs[i//4, i%4].xcorr(x, y, normed=normed, usevlines=usevlines, maxlags=max_lags, lw=xcorr_lw, detrend=mlab.detrend_mean)
            axs[i//4, i%4].fill_between(x_threshold, y_upper_max, y_lower_max, color=color_fillbetween, alpha=alpha_fillbetween)
            # Plot aestethics
            axs[i//4, i%4].set_title(x_variable, fontsize=fontsize_sub_title)
            axs[i//4, i%4].set_xlabel('<-- lag | lead -->', fontsize=fontsize_label)
            axs[i//4, i%4].grid(axis='x')
            axs[i//4, i%4].set_xticks(np.arange(-max_lags, max_lags+5, n_x_ticks))
            axs[i//4, i%4].tick_params(axis='x', labelbottom=True)
            
        # Disable any unused or empty plots
        i += 1
        while i < axs.size: 
            axs[i//4, i%4].set_visible(False)
            i += 1
        
        # Layout and plot    
        fig.suptitle(f"Cross Correlation Against {y_variable.title()}", fontsize=fontsize_title)
        fig.tight_layout(rect=rect)
        
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"cross_correlation_all_{y_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def single_correlate_plot(self, y_variable: str, x_variable: str, max_lags=30, streamlit = False, save_path=None, figsize=(15,6), dpi=180, **kwargs):
        """Function to plot a correlation plot between  variable y and x for n lags.

        Args:
            y_variable (str): Column name of the target variable in the dataframe.
            x_variable (str): Column name of the feature variable in the dataframe.
            max_lags (int, optional): The maximum number of lags that are used for the correlation plot. Defaults to 30.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (15,6).
            dpi (int, optional): dpi (int, optional): DPI value of the plot. Defaults to 180.
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 14
        fontsize_xyticks = kwargs['fontsize_xyticks'] if kwargs.get('fontsize_xyticks') else 12
        n_x_ticks = kwargs["n_x_ticks"] if kwargs.get("n_x_ticks") else 10
        threshold_value = kwargs["threshold_value"] if kwargs.get("threshold_value") else 0.1
        color_fillbetween = kwargs["color_fillbetween"] if kwargs.get("color_fillbetween") else "pink"
        alpha_fillbetween = kwargs["alpha_fillbetween"] if kwargs.get("alpha_fillbetween") else 0.2
        xcorr_lw  = kwargs["xcorr_lw"] if kwargs.get("xcorr_lw") else 5
        usevlines  = kwargs["usevlines"] if kwargs.get("usevlines") else True
        normed  = kwargs["normed"] if kwargs.get("normed") else True
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        
        # Generate Plots
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
        # Generate threshold dataset
        x_threshold = np.arange(-max_lags, max_lags+10, n_x_ticks)
        y_upper_max = [threshold_value]*len(x_threshold)
        y_lower_max = [-threshold_value]*len(x_threshold)
        
        x, y = self.df[x_variable].fillna(0), self.df[y_variable].dropna()
        ax.xcorr(x, y, normed=normed, usevlines=usevlines, maxlags=max_lags, lw=xcorr_lw, detrend=mlab.detrend_mean)
        ax.fill_between(x_threshold, y_upper_max, y_lower_max, color=color_fillbetween, alpha=alpha_fillbetween)
        
        # Plot aestethics
        ax.set_title(f"{y_variable.title()} vs {x_variable.title()}", fontsize=fontsize_title)
        ax.set_xlabel('<-- lead | lag -->', fontsize=fontsize_label)
        ax.set_xticks(np.arange(-max_lags, max_lags+5, n_x_ticks))
        ax.tick_params(axis='x', labelbottom=True)
        ax.tick_params(axis = 'both', labelsize=fontsize_xyticks)
        
        fig.tight_layout()
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"cross_correlation_{y_variable}_v_{x_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()
        
    def single_granger_plot(self, y_variable: str, x_variable: str, max_lags=30, streamlit = False, save_path=None, figsize=(15,6), dpi=180, **kwargs):
        """ Function to plot the granger causality between x and y for n lags.

        Args:
            y_variable (str): Name of the target variable
            x_variable (str): Name of the feature variable
            max_lags (int, optional): Number of max lags applied in the granger function. Defaults to 30.
            save_path (str, optional): Optional save path for a .png image of the plot. Should be direct path. Defaults to None.
            figsize (tuple, optional): Figure size of the plot in inch. Defaults to (15,6).
            dpi (int, optional): dpi (int, optional): DPI value of the plot. Defaults to 180.
        """
        # Parse some kwargs configurations
        fontsize_title = kwargs['fontsize_title'] if kwargs.get('fontsize_title') else 20
        fontsize_label = kwargs['fontsize_label'] if kwargs.get('fontsize_label') else 14
        fontsize_xyticks = kwargs['fontsize_xyticks'] if kwargs.get('fontsize_xyticks') else 12
        file_name_addition = kwargs["file_name_addition"] if kwargs.get("file_name_addition") else "" # add any additional string to the file name.
        transparent = kwargs["transparent"] if kwargs.get("transparent") else False # Set to false since facecolor is set to default. Would overwrite facecolor to make transparent.
        facecolor = kwargs["facecolor"] if kwargs.get("facecolor") else "#151934"
        show_pval = kwargs["show_pval"] if kwargs.get("show_pval") else True # If p-value is shown. Can be True and False.
        
        # Filter data:
        x, y = self.df[x_variable].fillna(0), self.df[y_variable].dropna()
        
        # Calculate Granger Causality
        lag_range = range(1, max_lags + 1)
        f_list = []
        p_list = []
        for lag in lag_range:
            res = grangercausalitytests(pd.DataFrame(x.dropna()).join(y.dropna(), how="inner"), maxlag=max_lags, verbose=False)
            f, p, _, _ = res[lag][0]["ssr_ftest"]
            f_list.append(f)
            p_list.append(p)
        
        # Generate Plots
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        ax.bar(x=lag_range, height=f_list)
        if show_pval:
            props = dict(boxstyle='round', facecolor='black', alpha=0.5)
            ax.text(x=0.5, y=0.8, s=f"minimum p-value = {min(p_list):.3f}", transform=ax.transAxes, bbox=props)
        
        # Plot aestethics
        ax.set_title(f"Granger causality, {y_variable.title()} vs {x_variable.title()}", fontsize=fontsize_title)
        ax.set_xlabel('lag -->', fontsize=fontsize_label)
        ax.set_ylabel("Granger Score (F)", fontsize=fontsize_label)
        ax.set_xticks(list(lag_range))
        ax.set_xticklabels(list(lag_range))
        ax.tick_params(axis='x', labelbottom=True)
        ax.tick_params(axis = 'both', labelsize=fontsize_xyticks)
        
        sns.despine()
        fig.tight_layout()
        if save_path!=None:
            fig.savefig(os.path.join(save_path, f"granger_causality_{y_variable}_v_{x_variable}{file_name_addition}"+".png"), facecolor=facecolor, transparent=transparent)
            plt.show()
        if streamlit == True:
            return fig
        
        plt.show()