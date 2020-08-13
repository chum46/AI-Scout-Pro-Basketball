import pandas as pd
import numpy as np
import seaborn as sns


def clean_oldNBA (df, columns):
    df.drop(['player_id', 'name_common', 'year_id', 'team_id', 'franch_id'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df