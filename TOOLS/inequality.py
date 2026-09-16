import numpy as np
import pandas as pd

def variable_distribution(values, distrib, label):
    """Returns a dataframe with all the possible values of a variable label and the associated frequency of agents"""
    assert np.size(values) == np.size(distrib), 'Requires same information for both variables'
    data = {label: values.flatten(), 'Share of agents': distrib.flatten()}
    df   = pd.DataFrame.from_dict(data)
    df   = df.sort_values(by=label)
    df['Cumulative share'] = df['Share of agents'].cumsum()
    return df

def variable_deciles(values, distrib, label, tol=1e-10):
    assert np.size(values) == np.size(distrib), 'Requires same information for both variables'
    df = variable_distribution(values, distrib, label)
    stats = {'Min': df.loc[0][label]}
    for i in range(1,10):
        stats['D{}'.format(i)] = df.iloc[np.sum(df['Cumulative share'] < i/10)][0]
    stats['Max'] = df[~np.isclose(df['Share of agents'], 0, atol=tol)].iloc[-1][label]
    return pd.DataFrame(stats, index=[label]).T