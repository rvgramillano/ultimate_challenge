import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import timedelta
from datetime import datetime
import calendar

# PART 1 - EXPLORATORY DATA ANALYSIS


def plot_daily_logins(df, counts):
    """Plot 15-min aggregated login counts for a single day.

    Takes in a login-counts dataframe for a single day,
    plots the data for that data
    and reformats the ticks:

    Args:
        df (pandas Dataframe): Single-day dataframe of login counts.
        counts (list): List of login counts from df.

    Returns:
        None: Writes to plot object.
    """
    # make counter variable for transitioning between days
    counter = 0
    # list for new time x-values
    times = []
    # loop over 15-min intervals for single-day df
    for time in df.index:
        '''this if-else aligns the labels between all days in the data set
         because the keep increasing,
         and we would like all days to be aligned by the hour.
         the days start at hour 20 and there are fifteen intervals
         between the first and second days.
         then the hour restarts at 0
         '''
        if time.hour <= 20 and counter >= 15:
            str_time = '02 ' + str(time)[11:16]
        else:
            str_time = '01 ' + str(time)[11:16]
        # get date into datetime, then to string with AM PM
        new_time = datetime.strptime(str_time, '%d %H:%M').strftime('%d %I:%M %p')
        # add to x-values
        times.append(new_time)
        counter += 1
    '''plot the curent day's login counts
    and alter tick labels to not include day 1 or 2
    and limit the number of ticks
    '''
    plt.plot(times, counts, color='cornflowerblue', alpha=.2)
    plt.xticks(times[::15], [t[3:] for t in times][::15], fontsize=9)
    return None


def create_avg_day_plot(df):
    """Create average day plot, with all day's login counts along with average
    plotted.

    Takes in a 15-min aggregated login counts dataframe and plots
    the login counts for each day along with average:

    Args:
        df (pandas Dataframe): Dataframe of 15-min aggregated login counts for
        all days.

    Returns:
        None: Saves a plot.
    """
    total_days = (df.index[-1] - df.index[0]).days
    # where we will store averaged login counts across all days
    all_days_avg = []
    starting_index = df.index[0]
    # loop over all days
    for day in range(total_days):
        # get first day of data and plot it
        cur_day = df.loc[starting_index:starting_index + timedelta(days=1)]
        # get 15-min aggregated login counts for that day,
        # plot that day's data and append to average list
        counts = cur_day.values
        plot_daily_logins(cur_day, counts)
        all_days_avg.append(counts)
        # increment a day
        starting_index = starting_index + timedelta(days=1)
    # get average over all days and plot it along with final plot options
    all_day_avg = np.array(np.mean(np.array(all_days_avg), axis=0))
    plt.plot(all_day_avg, color='black', label='Average', linewidth=2.5)
    plt.title('15-min Aggregated Login Counts Averaged Across All Days', fontsize=12)
    plt.ylabel('Login Counts')
    plt.xlabel('Time of Day')
    plt.legend()
    plt.grid(linestyle=':')
    plt.savefig(DIR + 'login_counts_day_avg.png', dpi=350)
    plt.cla()
    return None


DIR = '/Users/rvg/Documents/springboard_ds/ultimate_challenge/'

# Load json file and put into pandas DataFrame
with open(DIR + 'logins.json') as f:
    data = json.load(f)

df = pd.DataFrame.from_records(data)

# Convert login times to datetime and make index col
df['login_time'] = pd.to_datetime(df['login_time'], format='%Y-%m-%d %H:%M:%S')

df.set_index('login_time', inplace=True)

# add count column for each login time for aggregation
df['count'] = np.ones(len(df))

# aggregated dataframe, aggregating login counts for 15 minute intervals
resampled_df = df.resample('15Min').sum()

# generate daily login-count plot
create_avg_day_plot(resampled_df)

# now we generate a bar plot of logins grouped by weekday.
resampled_df['weekday'] = resampled_df.index.weekday
ax = (resampled_df.groupby('weekday')['count'].sum()).plot(kind='bar')
plt.title("Number of Logins per Weekday")
plt.xlabel("Weekday")
plt.ylabel("Number of logins")
plt.xticks(rotation=45)
ax.set_xticklabels([calendar.day_name[d] for d in range(7)])
plt.tight_layout()
plt.savefig(DIR + 'login_counts_weekday.png', dpi=350)
