# Step 0: Create duration and amplitude arrays
# ------------------------------------------------------------------------
# Import Functions
# ------------------------------------------------------------------------
import numpy as np
from scipy import stats
from burst_functions import get_burst_features
from burst_functions import identify_bursts
import pandas as pd
# ------------------------------------------------------------------------
# Select analysis
# ------------------------------------------------------------------------

# standard parameters
days = list(range(1,19+1))
f_beta = (4,40)
plot_steps = 0
f_lowpass = 40
Fs = 781
for monkey in ['Satan', 'Risette']:
    if monkey == 'Risette':
        n_el = 22
    else:
        n_el = 29
    df = get_burst_features(monkey, days, f_beta, f_lowpass, Fs, df=None)
    df.to_pickle('34783_' + monkey +  '_df_' + str(f_beta))
