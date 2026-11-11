import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = 'Data'

def match_colleges(user_weights, constraints=None, selected_major=None):
    institution_file_path = os.path.join(DATA_DIR, 'institutions_clean.csv')
    df = pd.read_csv(institution_file_path)

    if constraints:
        for key, value in constraints.items():
            if key in df.columns:
                df = df[df[key] == value]

    if selected_major:
        programs_file_path = os.path.join(DATA_DIR, 'programs_clean.csv')
        programs_df = pd.read_csv(programs_file_path)

        matching_programs = programs_df[
            programs_df['major_name'].str.contains(selected_major, case=False, na=False)
        ].copy()
        valid_units = matching_programs['unit_id'].unique()
        df = df[df['unit_id'].isin(valid_units)]

        for column in ('median_earnings_1yr', 'median_federal_debt'):
            matching_programs[column] = pd.to_numeric(matching_programs[column], errors='coerce')
        program_metrics = matching_programs.groupby('unit_id')[
            ['median_earnings_1yr', 'median_federal_debt']
        ].median()
        df = df.merge(program_metrics, on='unit_id', how='left')

    if df.empty:
        return pd.DataFrame() # no matches to constraints

    scaler = MinMaxScaler()
    df[['norm_price', 'norm_grad', 'norm_size']] = scaler.fit_transform(
        df[['net_price', 'graduation_rate', 'undergrad_size']]
    )

    w = np.array(user_weights)
    ideal_vector = np.array([0.0, 1.0, 0.5]) # ideal values for price (low), graduation rate (high), and size (medium)
    user_vector = ideal_vector * w
    college_matrix = df[['norm_price', 'norm_grad', 'norm_size']].values * w

    similarities = cosine_similarity(college_matrix, user_vector.reshape(1, -1)).flatten()
    df['match_score'] = similarities * 100 # Convert to percentage

    df['roi'] = np.nan
    df['roi_score'] = np.nan
    df['ranking_score'] = df['match_score']
    if selected_major:
        has_roi_data = (
            df['median_earnings_1yr'].notna()
            & df['median_federal_debt'].notna()
            & (df['median_federal_debt'] >= 0)
            & (
                (df['median_federal_debt'] > 0)
                | (df['median_earnings_1yr'] > 0)
            )
        )
        roi = pd.Series(np.nan, index=df.index)
        roi_score = pd.Series(50.0, index=df.index)
        if has_roi_data.any():
            roi_values = (
                df.loc[has_roi_data, 'median_earnings_1yr']
                / df.loc[has_roi_data, 'median_federal_debt'].replace(0, np.nan)
            )
            roi_values = roi_values.mask(
                df.loc[has_roi_data, 'median_federal_debt'] == 0, np.inf
            )
            roi.loc[has_roi_data] = roi_values
            roi_score.loc[has_roi_data] = roi_values.rank(pct=True) * 100

        df['roi'] = roi
        df['roi_score'] = roi_score
        df['ranking_score'] = 0.5 * df['match_score'] + 0.5 * df['roi_score']

    return df.sort_values(
        by=['ranking_score', 'match_score'], ascending=False
    ).head(10)