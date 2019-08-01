## Step 1: Descriptive of bursts
# How often do they occur,  what do they look like, what are their features

# ------------------------------------------------------------------------
# import
# ------------------------------------------------------------------------
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------------------
# Select analysis
# ------------------------------------------------------------------------
monkey = 'Satan'                # 'Risette'
f_beta = (4, 40)
if monkey == 'Risette':
    n_el = 22
else:
    n_el = 29

# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
df = pd.read_pickle('34783_'+ monkey +'_df_' + str(f_beta))
df = df.reset_index()
for d in set(df['day']):
    df_day = df[df['day'] == d]
    if d == list(set(df['day']))[0]:
        n_trials_per_day = [len(set(df_day['trial']))]
    else:
        n_trials_per_day.append(len(set(df_day['trial'])))

# ------------------------------------------------------------------------
# Convert samples into seconds
# ------------------------------------------------------------------------
df['duration_ms'] = df['duration']*1000/781
df['onset_ms'] = df['onset']*1000/781
df['end_ms'] = df['end']*1000/781

# ------------------------------------------------------------------------
# How does a burst look like
# ------------------------------------------------------------------------
# (plot_trial/burst/violation functions are broken due to the new length of data)
el = 5
d = 1
from burst_functions import plot_burst
from burst_functions import plot_burst_violations
phase = 'repeat'
for trial in range(1,10):
    plot_burst(monkey, d, el, trial, f_beta, True)
    #plot_burst_violations(monkey, d, conditions, data, el, trial, str((4, 10)))

# ------------------------------------------------------------------------
# relative frequency of trials with at least one burst
# ------------------------------------------------------------------------

rel_freq_burst_tmp = np.zeros(shape=(n_el, 2))
for el in range(n_el):
    df_el = df[df['electrode'] == el]
    if el == 0:
        rel_freq_burst = [1 - sum(df_el['duration'].isna())/sum(n_trials_per_day)]
    else:
        rel_freq_burst.append(1- sum(df_el['duration'].isna())/sum(n_trials_per_day))


# plot relative frequency for each electrode
x = list(range(1,n_el+1))
fig, ax = plt.subplots()
ax.bar(x,rel_freq_burst, align='center', alpha=0.5, ecolor='black',
       capsize=10)
ax.set_ylabel('Relative Frequency')
ax.set_xlabel('Electrodes')
ax.set_xticks(x)
ax.set_title('Relative Frequency per Electrode')
ax.yaxis.grid(True)
#ax.set_ylim(0.8, 1)


# ------------------------------------------------------------------------
# Relative burst frequency averaged across electroes
# ------------------------------------------------------------------------
mean_occurrence = round(np.mean(rel_freq_burst),3)
std_occurrence = round(np.std(rel_freq_burst)/math.sqrt(n_el), 3)
print('Average Burst frequency for ' + monkey + ': ' + str(mean_occurrence) + ' (' + str(std_occurrence) +')')

# ------------------------------------------------------------------------
# distribution of counts of bursts per trial
# ------------------------------------------------------------------------
count_burst = np.zeros(shape=(n_el, 7))
for el in set(df['electrode']):
    df_el = df[df['electrode'] == el]
    for d in set(df_el['day']):
        df_el_day = df_el[df_el['day'] == d]
        for trial in set(df_el_day['trial']):
            df_el_day_trial = df_el_day[df_el_day['trial'] == trial]
            # 0 bursts
            if df_el_day_trial['duration'].isna().any():
                n_bursts = 0
            else:
                n_bursts = len(df_el_day_trial)
            count_burst[el, n_bursts] += 1
# sanity check: sum of n_bursts is the same as n_trials
np.sum(count_burst, axis = 1) == sum(n_trials_per_day)
# relative frequency of nbursts
distribution_nbursts = count_burst / sum(n_trials_per_day)

# plot
x = list(range(5))
fig, ax = plt.subplots()
mean_occ = np.mean(distribution_nbursts, axis = 0)[:5]
se_occ = (np.std(distribution_nbursts, axis = 0)/math.sqrt(n_el))[:5]
ax.bar(x,mean_occ, yerr=se_occ, align='center', alpha=0.5, ecolor='black',
       capsize=10)
ax.set_ylabel('Relative Frequency')
ax.set_xlabel('Burst Order')
ax.set_xticks(x)
ax.set_title('Relative Frequency per Burst Count')
ax.yaxis.grid(True)
#ax.set_ylim(0, 0.65)

# ------------------------------------------------------------------------
# Are there any electrodes for which the distribution of number of bursts deviates?
# ------------------------------------------------------------------------
distribution_diff = distribution_nbursts - np.mean(distribution_nbursts, axis = 0)
for el, el_dis in enumerate(distribution_diff):
    if sum(el_dis > 0.05) > 0:
        print(el)


# ------------------------------------------------------------------------
# Duration Descriptives
# ------------------------------------------------------------------------
# Distribution of duration per electrode
# boxplot
df['duration_ms'] = df['duration_ms'].astype(float)
sns.catplot(x="electrode", y="duration_ms", kind="box", data=df.dropna())
plt.title('Duration per channel for ' + monkey , y=1.1, fontsize=18)

# duruation per burst
sns.catplot(x="burst_index", y="duration_ms", kind="box", data=df.dropna())
plt.title('Duration per burst for ' + monkey , y=1.1, fontsize=18)


# ------------------------------------------------------------------------
# Amplitude Descriptives
# ------------------------------------------------------------------------
# Per electrode
df['amplitude_med'] = df['amplitude_med'].astype(float)
sns.catplot(x="electrode", y="amplitude_med", kind="box", data=df.dropna())
plt.title('Median Amplitude per channel for ' + monkey , y=1.1, fontsize=18)
# Per Burst order
sns.catplot(x="burst_index", y="amplitude_med", kind="box", data=df.dropna())
plt.title('Median Amplitude per Burst Rank for ' + monkey , y=1.1, fontsize=18)


df['amplitude_avg'] = df['amplitude_avg'].astype(float)

sns.catplot(x="burst_index", y="amplitude_avg", kind="box", data=df)
plt.title('Average Amplitude per Burst Rank for ' + monkey , y=1.1, fontsize=18)
sns.catplot(x="electrode", y="amplitude_avg", kind="box", data=df.dropna())
plt.title('Mean Amplitude per channel for ' + monkey , y=1.1, fontsize=18)


# relationship duration ~ ampltude
g = sns.FacetGrid(df, col="electrode", col_wrap=5)
g.map(plt.scatter, "duration_ms", "amplitude_med", edgecolor="white", s=50, lw=1, alpha = 0.7)

# ------------------------------------------------------------------------
## Timing of burst
# ------------------------------------------------------------------------
df_el = df.dropna()[df['electrode'] == 1]
burst_per_time_sample = np.zeros(shape = (1500,1))
for index, row in df_el.iterrows():
    burst_per_time_sample[int(round(row['onset_ms'])):int(round(row['end_ms']))] += 1

x = list(range(499,1500))
fig, ax = plt.subplots()
ax.plot(burst_per_time_sample[499:1500])

# ------------------------------------------------------------------------
# period/ frequency
# ------------------------------------------------------------------------
g = sns.FacetGrid(df, col="electrode", col_wrap=5)
g.map(sns.distplot, "duration_ms")

g = sns.FacetGrid(df, col="electrode", col_wrap=5)
g.map(sns.distplot, "amplitude_med")
df['frequency'] = [int(round(1/(i/ 1000))) if not math.isnan(i) else float('nan') for i in df['period_med']]
g = sns.FacetGrid(df, col="electrode", col_wrap=5)
g.map(sns.distplot, "frequency")

