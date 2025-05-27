import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_combined_lineplot(df, group_by='year', hue='MU', output_path=None):
    """
    Create a bar plot of percent positive volume.
    """
    plt.figure(figsize=(14, 8))

    # Calculate mean values for each group
    summary_df = df.groupby([group_by, hue])['percent_positive'].mean().reset_index()

    # Sort if using month as group_by
    if group_by == 'month':
        # Define month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Filter to only include months that are present in the data
        month_order = [m for m in month_order if m in summary_df['month'].unique()]

        # Create bar plot with proper month ordering
        ax = sns.lineplot(x=group_by, y='percent_positive', hue=hue, data=summary_df, order=month_order)
    else:
        # For other group_by values, no special ordering needed
        ax = sns.lineplot(x=group_by, y='percent_positive', hue=hue, data=summary_df)

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title(f'Average Percent Volume with Positive GRP Values by {group_by}')
    plt.xlabel(group_by.capitalize())
    plt.ylabel('Percent Positive Volume (%)')
    plt.legend(title=hue)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.close()

def plot_combined_relplot(df, group_by='year', hue='MU', output_path=None):
    """
    Create a bar plot of percent positive volume.
    """
    plt.figure(figsize=(14, 8))

    # Calculate mean values for each group
    summary_df = df.groupby([group_by, hue])['percent_positive'].mean().reset_index()

    # Sort if using month as group_by
    if group_by == 'month':
        # Define month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Filter to only include months that are present in the data
        month_order = [m for m in month_order if m in summary_df['month'].unique()]

        # Create bar plot with proper month ordering
        ax = sns.relplot(data=summary_df, x=group_by, y='percent_positive', hue=hue, row = 'year', col = 'GRP_type', order=month_order, kine = 'line')
    else:
        # For other group_by values, no special ordering needed
        ax = sns.relplot(data=summary_df, x=group_by, y='percent_positive', hue=hue, row ='year', col ='GRP_type', kine = 'line')

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title(f'Average Percent Volume with Positive GRP Values by {group_by}')
    plt.xlabel(group_by.capitalize())
    plt.ylabel('Percent Positive Volume (%)')
    plt.legend(title=hue)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    plt.close()


# load .csv as df

mu_ts = pd.read_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Ch2_dfs&plots/complete_GRP_volume_data.csv')


plot_combined_relplot(mu_ts, group_by='year', hue='MU') # row = 'year', col = 'GRP_type')


#%%
# check
print(mu_ts.describe())

# set numeric month index
mu_ts.set_index('month_num')

# line groups by MU
mu_groups = mu_ts.groupby('MU')

# subplots are years
yr_groups = mu_ts.groupby("year")

# grp_type groups
grp_tgroups = mu_ts.groupby("GRP_type")

#grp_tgroups = mu_ts.groupby("GRP")
ncols= 4
nrows = 3 #int(np.ceil(mu_groups/ncols))

hue_tst = mu_ts['MU']
# Define month order
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Filter to only include months that are present in the data
month_order = [m for m in month_order if m in mu_ts['month'].unique()]

huemu_dict = {k:v for k,v in mu_ts['MU'].items()}

sns.relplot(kind='line', data=mu_ts, x='month', y='percent_positive', hue='MU', row= 'year', col='GRP_type')
# Set y-limits from 0 to 100 for percentage
plt.ylim(0, 100)
plt.title('Average Percent Volume with Positive GRP Values by')
plt.xlabel("month number")
plt.ylabel('Percent Positive Volume (%)')
plt.legend(huemu_dict)
plt.tight_layout()
plt.show()

#plt.legend()

#plot_combined_lineplot(mu_ts, group_by='month', hue='MU', row = 'year'[0:8], col= 'year'[9:16], output_path=None)
#plot_combined_lineplot(mu_ts, group_by='year', hue='MU', output_path=None)


#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import numpy as np

"should be optimal here with the most configs/options "


def plot_combined_lineplot(df, group_by='year', hue='MU', output_path=None):
    """
    Create a line plot of percent positive volume.
    """
    plt.figure(figsize=(14, 8))

    # Calculate mean values for each group
    summary_df = df.groupby([group_by, hue])['percent_positive'].mean().reset_index()

    # Sort if using month as group_by
    if group_by == 'month':
        # Define month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Filter to only include months that are present in the data
        month_order = [m for m in month_order if m in summary_df['month'].unique()]

        # Create bar plot with proper month ordering
        ax = sns.lineplot(x=group_by, y='percent_positive', hue=hue, data=summary_df, order=month_order)
    else:
        # For other group_by values, no special ordering needed
        ax = sns.lineplot(x=group_by, y='percent_positive', hue=hue, data=summary_df)

    # Set y-limits from 0 to 100 for percentage
    plt.ylim(0, 100)

    plt.title(f'Average Percent Volume with Positive GRP Values by {group_by}')
    plt.xlabel(group_by.capitalize())
    plt.ylabel('Percent Positive Volume (%)')

    # Improve legend
    plt.legend(title=hue, loc='best', frameon=True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)

    return plt.gcf()


def plot_combined_relplot(df, group_by='month', hue='MU', row='year', col='GRP_type',
                          n_rows=None, n_cols=None, output_path=None):
    """
    Create a relational plot with flexible grid structure for facets.

    Parameters:
    -----------
    df : pandas DataFrame
        Data to plot
    group_by : str
        Column to use for x-axis
    hue : str
        Column to use for color coding
    row : str or None
        Column to use for row facets
    col : str or None
        Column to use for column facets
    n_rows : int or None
        Number of rows for the plot grid (overrides automatic layout)
    n_cols : int or None
        Number of columns for the plot grid (overrides automatic layout)
    output_path : str or None
        Path to save the figure

    Returns:
    --------
    g : seaborn.FacetGrid
        The resulting plot
    """
    # Calculate mean values for each group
    groupby_cols = [col for col in [group_by, hue, row, col] if col is not None]
    summary_df = df.groupby(groupby_cols)['percent_positive'].mean().reset_index()

    # Set up facet grid dimensions
    facet_kws = {}
    if n_rows is not None:
        facet_kws['row_order'] = sorted(df[row].unique()) if row else None
    if n_cols is not None:
        facet_kws['col_order'] = sorted(df[col].unique()) if col else None

    # Sort if using month as group_by
    if group_by == 'month':
        # Define month order
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        # Filter to only include months that are present in the data
        month_order = [m for m in month_order if m in summary_df['month'].unique()]

        # Create relplot with proper month ordering
        g = sns.relplot(
            data=summary_df,
            x=group_by,
            y='percent_positive',
            hue=hue,
            row=row,
            col=col,
            kind='line',
            height=3.5,
            aspect=1.2,
            facet_kws=facet_kws,
            palette="colorblind",  # Use a colorblind-friendly palette
            #order=month_order
        )
    else:
        # For other group_by values, no special ordering needed
        g = sns.relplot(
            data=summary_df,
            x=group_by,
            y='percent_positive',
            hue=hue,
            row=row,
            col=col,
            kind='line',
            height=3.5,
            aspect=1.2,
            facet_kws=facet_kws,
            palette="colorblind"  # Use a colorblind-friendly palette
        )

    # Set y-limits from 0 to 100 for percentage
    g.set(ylim=(0, 100))

    # Set titles and labels
    g.set_axis_labels(f"{group_by.capitalize()}", "Percent Positive Volume (%)")
    g.fig.suptitle(f'Average Percent Volume with Positive GRP Values by {group_by}', fontsize=16)
    g.fig.subplots_adjust(top=0.9)  # Adjust to make room for the title

    # Improve the legend - make it more visible and descriptive
    g.add_legend(title=hue, frameon=True, bbox_to_anchor=(1.05, 0.5), loc='center left')

    # Adjust the figure size based on number of facets if specified
    if n_rows is not None and n_cols is not None:
        plt.gcf().set_size_inches(n_cols * 4, n_rows * 3)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    return g


# Usage example
if __name__ == "__main__":
    # Load .csv as df
    # Replace this path with your actual file path
    #mu_ts = pd.read_csv('/Users/rajeevkumar/Documents/Labs/HöökLab/Ch2_dfs&plots/complete_GRP_volume_data.csv')
    file_path = '/Users/rajeevkumar/Documents/Labs/HöökLab/Ch2_dfs&plots/complete_GRP_volume_data.csv'
    mu_ts = pd.read_csv(file_path)
    ncols =1
    nrows = 1

    # Example 1: Basic relplot with default grid layout
    g1 = plot_combined_relplot(
        mu_ts,
        group_by='month',
        hue='MU',
        row='year',
        col='GRP_type',
        output_path='/Users/rajeevkumar/Documents/Labs/HöökLab/Ch2_dfs&plots/relplot_default_grid.png'
    )
    plt.close()

    # Example 2: Custom grid layout with 2 rows, 3 columns
    g2 = plot_combined_relplot(
        mu_ts,
        group_by='month',
        hue='MU',
        row='year',
        col='GRP_type',
        n_rows=None,
        n_cols=None,
        output_path='/Users/rajeevkumar/Documents/Labs/HöökLab/Ch2_dfs&plots/relplot_custom_grid.png'
    )
    plt.close()

    # Example 3: Simple lineplot for comparison
    fig = plot_combined_lineplot(
        mu_ts,
        group_by='month',
        hue='MU',
        output_path='/Users/rajeevkumar/Documents/Labs/HöökLab/Ch2_dfs&plots/lineplot_tst.png'
    )
    plt.close()

    print("Plots generated successfully!")