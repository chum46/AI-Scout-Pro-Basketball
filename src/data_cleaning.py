import pandas as pd
import numpy as np
import seaborn as sns


def clean_old_df (df, columns):
    df.drop(['player_id', 'name_common', 'year_id', 'team_id', 'franch_id'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

def clean_mod_df (df, columns):
    df.drop(['player_id', 'name_common', 'year_id', 'team_id', 'franch_id'], axis=1, inplace=True)
    df['3P%'].fillna(0, inplace = True)
    millemiFT = df.loc[df_newNBA['name_common'] == 'Mike Miller', 'FT%'].mean()
    rudezdaFT = 69.6
    df.at[1460,'FT%'] = millemiFT
    df.at[1272,'FT%'] = rudezdaFT
    return df