
def get_data(monkey, d):
    '''
    :param monkey: 'Satan' or 'Risette'
    :param d: integer starting at 1
    :return:
    '''
    import scipy as sp
    import scipy.io

    fname = '/Volumes/Elements/1.Ghent/Cognitive_Control/burst_test/data_complete/' + monkey + '_c0_d_' + str(
        d) + '.mat'
    data_load = sp.io.loadmat(fname)
    data = data_load['data_C']
    return data


def identify_bursts(monkey, days, f_beta, f_lowpass, Fs):
    '''
    :param monkey: Satan or Risette
    :param days: range of days, starts at 1
    :param f_beta: frequency range of interest, e.g. (4,40)
    :param f_lowpass: integer for lowpass filter, e.g. 40
    :param Fs: samplting frequency
    :param n_bins: number of bins for time analysis, e.g. 8
    :return:
    '''
    # ------------------------------------------------------------------------
    # 1. import packages
    # ------------------------------------------------------------------------
    import numpy as np
    from bycycle.filt import lowpass_filter
    import pandas as pd
    pd.options.display.max_columns = 50
    from bycycle.filt import bandpass_filter
    from bycycle.cyclepoints import _fzerorise, _fzerofall, find_extrema
    from bycycle.features import compute_features
    # ------------------------------------------------------------------------
    #  Define parameters
    # ------------------------------------------------------------------------
    length_trial = 1564
    # ------------------------------------------------------------------------
    # 2. load and select data
    # ------------------------------------------------------------------------
    df_all = []
    for d in days:
        print('Identifying bursts for day', d)
        data = get_data(monkey, d)
        n_trials = data.shape[1] // length_trial
        n_el = data.shape[0]
        # ------------------------------------------------------------------------
        # 3. Prepare data
        # ------------------------------------------------------------------------
        for el in range(n_el):
            # initialize condition array per electrode
            signals_raw = []  # will contain all trials as array
            # separate into trials
            for trial in range(n_trials):
                signals_raw.append(data[el, trial * length_trial: ((trial + 1) * length_trial)])
            # lowpass filter
            # aim: remove slow transients or high frequency activity that may interfer with peak and trough identification
            signals_filtered = []
            for signals_raw_trial in signals_raw:
                signals_filtered.append(
                    lowpass_filter(signals_raw_trial, Fs, f_lowpass, N_seconds=.2, remove_edge_artifacts=False))
        # ------------------------------------------------------------------------
        # 4. cycle feature computation
        # ------------------------------------------------------------------------
            burst_kwargs = {'amplitude_fraction_threshold': .3,
                            'amplitude_consistency_threshold': .4,
                            'period_consistency_threshold': .5,
                            'monotonicity_threshold': .8,
                            'N_cycles_min': 3}
            dfs = []
            for trial, signal_filtered_narrow_trial in enumerate(signals_filtered):
                df = compute_features(signal_filtered_narrow_trial, Fs, f_beta, burst_detection_kwargs=burst_kwargs)
                # only take bursts that are within 1000 ms delay
                df = df.loc[(df['sample_last_trough'] >= 392) & (df['sample_next_trough'] <= 1172)]
                df = df.reset_index()
                df['trial'] = trial + 1  # Search
                df['electrode'] = el
                df['day'] = d
                dfs.append(df)
            df_cycles = pd.concat(dfs)  # make panda df
            df_all.append(df_cycles)
    df_all = pd.concat(df_all)
    df_all.to_pickle(monkey +  '_dfBurstRaw_' + str(f_beta))
    return df_all


def get_burst_features(monkey, days, f_beta, f_lowpass, Fs, n_bins, df=None):
    '''
    :param monkey: Satan or Risette
    :param days: range of days, starts at 1
    :param f_beta: frequency range of interest, e.g. (4,40)
    :param f_lowpass: integer for lowpass filter, e.g. 40
    :param Fs: sampling frequency
    :param n_bins: number of bins for time analysis, e.g. 8
    :param df: dfBurstRaw in case it's already available
    :return:
    '''
    import numpy as np
    import pandas as pd
    if df is None:
        df = identify_bursts(monkey, days, f_beta, f_lowpass, Fs, n_bins)
    colNames = ['monkey', 'day', 'electrode', 'trial', 'burst_index', 'onset', 'duration', 'end', 'amplitude_avg',
                'amplitude_med', 'period_med']
    df_burst_features = pd.DataFrame(columns = colNames)
    for el in set(df['electrode']):
        print('Compiling burst features for electrode ', el)
        for d in set(df['day']):
            for trial in set(df['trial']):
                df_subset = df[(df['electrode'] == el) & (df['day'] == d) & (df['trial'] == trial)]
                df_subset['burst_index'] = 0
                df_subset['burst_period_index'] = 0
                index_burst_period = 0
                index_cycle_of_burst = 0

                for b in range(len(df_subset)):
                    # burst
                    if df_subset.ix[b]['is_burst']:  # identify burst
                        if b == 0:   # first cycle
                            index_cycle_of_burst = 1
                            index_burst_period += 1
                        else:
                            # consecutive burst
                            if df_subset.ix[b - 1]['is_burst']:  # if previous is also burst
                                index_cycle_of_burst += 1
                            # first burst
                            elif not df_subset.ix[b - 1]['is_burst']:  # first burst
                                index_burst_period += 1
                                index_cycle_of_burst = 1
                    # no burst
                    else:
                        # ende of burst
                        if b > 0:
                            if df_subset.ix[b - 1]['is_burst']:  # if previous is burst
                                index_cycle_of_burst = 0
                    df_subset.at[b, 'burst_index'] = index_cycle_of_burst
                    df_subset.at[b, 'burst_period_index'] = index_burst_period


                ## Create burst df for a trial
                # if no burst in trial
                how_many_bursts = len([i for i in list(df_subset['burst_period_index'].unique()) if i > 0])

                if how_many_bursts == 0:
                    df_burst = pd.DataFrame(index=range(1), columns=colNames)
                    df_burst['monkey'][0] = monkey
                    df_burst['day'][0] = d
                    df_burst['trial'][0] = trial
                    df_burst['electrode'][0] = el
                else:
                    df_burst = pd.DataFrame(
                        index=range(how_many_bursts),
                        columns=colNames)

                # Calculate features of burst
                for i in range(how_many_bursts):
                    df_burst_i = df_subset[
                        np.logical_and(np.array(df_subset['burst_period_index'] == i+1),
                                       np.array(df_subset['burst_index'] != 0))]
                    df_burst_i = df_burst_i.set_index(
                        [pd.Index(list(range(1, len(df_burst_i) + 1)))])

                    # calculate features
                    onset = df_burst_i.loc[1, 'sample_last_trough']
                    end = df_burst_i.loc[len(df_burst_i), 'sample_next_trough']
                    duration = end - onset
                    amplitude_avg = np.mean(df_burst_i['volt_amp'])
                    amplitude_med = np.median(df_burst_i['volt_amp'])
                    period_med = np.median(df_burst_i['period'])
                    # add to dataframe
                    df_burst['monkey'][i] = monkey
                    df_burst['day'][i] = d
                    df_burst['trial'][i] = trial
                    df_burst['burst_index'][i] = i+1
                    df_burst['electrode'] = el
                    df_burst['onset'][i] = onset
                    df_burst['duration'][i] = duration
                    df_burst['end'][i] = end
                    df_burst['amplitude_avg'][i] = amplitude_avg
                    df_burst['amplitude_med'][i] = amplitude_med
                    df_burst['period_med'][i] = period_med
                df_burst_features = pd.concat([df_burst_features, df_burst])
    return df_burst_features
