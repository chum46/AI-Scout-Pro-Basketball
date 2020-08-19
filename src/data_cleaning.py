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

def clean_labeled_df (df):
    df['gmm_cluster'] = df['gmm_cluster'].map({0: 'c0', 1: 'c1', 2: 'c2', 3: 'c3', 4: 'c4', 5: 'c5'})
    df['class'] = df['class'].map({'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4})
    df.rename(columns={'class': 'class_'}, inplace=True)
    group_df = df[df['class_'] == df.groupby(['player_name','gmm_cluster'])['class_'].transform('max')]
    return group_df

def clean_future_df (df):
    df.set_index(['pid','player_name'], inplace = True)    
    future_df = df.drop(columns=['pick','year','pfr','ht','class','GP','conf','team'], axis = 1)
    future_df['3PAr'].fillna(0, inplace = True)
    future_df['ast/tov'].fillna(0, inplace = True)
    future_df.dropna(inplace=True)
    return future_df