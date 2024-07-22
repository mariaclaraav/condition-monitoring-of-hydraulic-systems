import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def tplot(X, title, ylabel):
    """
    Plots the time series data with statistical limits.

    Parameters:
    X (pd.DataFrame): DataFrame containing the time series data to be plotted.
    title (str): Title of the plot.
    ylabel (str): Label for the y-axis.
    """
    datanew = X.copy()
    flattened_data = datanew.to_numpy().flatten()
    Q1 = np.quantile(flattened_data, 0.25)
    Q3 = np.quantile(flattened_data, 0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(X, color='0.8', alpha=0.05)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Measurements', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust nbins to control the number of xticks

    ax.hlines(upper_limit, 0, len(X), label='Upper Limit', lw=1.5, color='tab:red')
    ax.hlines(lower_limit, 0, len(X), label='Lower Limit', lw=1.5, color='tab:purple')

    ax.legend(loc='upper right', fancybox=True, shadow=True, ncol=5, prop={'size': 12}),

    ax.grid(True)
    plt.show()
    
# descplot plotting function
def descplot(desc, title, y_axis_label):
    # Create a figure and axis object with a specific size
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Plot the median values
    ax.plot(desc['Median'], label='Median', color='tab:blue')
    
    # Fill the area between the first and third quartiles
    ax.fill_between(x=range(len(desc['Median'])), y1=desc['Q3'], y2=desc['Q1'], facecolor='tab:blue', alpha=0.3, label='Quartile')
    
    # Plot the maximum and minimum values
    ax.plot(desc['Max'], label='Maximum', color='tab:red')
    ax.plot(desc['Min'], label='Minimum', color='tab:purple')
    
    # Set the title and labels for the axes
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Measurements', fontsize=16)
    ax.set_ylabel(y_axis_label, fontsize=16)
    
    ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=5, prop={'size': 10})
    
    # Adjust the x-axis ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust nbins to control the number of xticks
    
    # Display the plot
    plt.grid(True)
    plt.show()
    
def descplot_2(desc, title, y_axis_label, X_profile):
    # Create a figure and axis object with a specific size
    fig, ax1 = plt.subplots(figsize=(14, 4))
    
    # Plot the median values
    ax1.plot(desc['Median'], label='Median', color='tab:blue')
    
    # Fill the area between the first and third quartiles
    ax1.fill_between(x=range(len(desc['Median'])), y1=desc['Q3'], y2=desc['Q1'], facecolor='tab:blue', alpha=0.3, label='Quartile')
    
    # Plot the maximum and minimum values
    ax1.plot(desc['Max'], label='Maximum', color='tab:red')
    ax1.plot(desc['Min'], label='Minimum', color='tab:purple')
    
    # Set the title and labels for the axes
    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('Measurements', fontsize=16)
    ax1.set_ylabel(y_axis_label, fontsize=16)
    
    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(X_profile['profile'], label='Profile', color='k', linestyle='--')
    ax2.set_ylabel('Profile', fontsize=16)
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fancybox=True, shadow=True, ncol=5, prop={'size': 10})
    
    # Adjust the x-axis ticks
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))  # Adjust nbins to control the number of xticks
    
    # Display the plot
    plt.grid(True)
    plt.show()