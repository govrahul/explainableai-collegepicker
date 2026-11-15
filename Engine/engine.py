import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = 'Data'

def match_colleges(
    user_weights,
    constraints=None,
    selected_major=None,
    ideal_size=0.5,
    prestige_weight=0.5,
):
    if (
        isinstance(prestige_weight, bool)
        or not isinstance(prestige_weight, (int, float, np.number))
        or not np.isfinite(prestige_weight)
        or not 0.0 <= prestige_weight <= 1.0
    ):
        raise ValueError('prestige_weight must be between 0.0 and 1.0')

    institution_file_path = os.path.join(DATA_DIR, 'institutions_clean.csv')
    df = pd.read_csv(institution_file_path)

    if constraints:
        numeric_bounds = {
            'min_graduation_rate': ('graduation_rate', 'min'),
            'max_graduation_rate': ('graduation_rate', 'max'),
            'min_median_earnings_10yr': ('median_earnings_10yr', 'min'),
            'max_median_earnings_10yr': ('median_earnings_10yr', 'max'),
            'min_median_debt_at_graduation': ('median_debt_at_graduation', 'min'),
            'max_median_debt_at_graduation': ('median_debt_at_graduation', 'max'),
            'min_repayment_within_3_years': ('repayment_within_3_years', 'min'),
            'max_repayment_within_3_years': ('repayment_within_3_years', 'max'),
            'max_average_sat_score': ('average_sat_score', 'max'),
            'max_average_act_score': ('average_act_score', 'max'),
        }
        for key, value in constraints.items():
            if key in ('state', 'states') and value is not None:
                states = [value] if isinstance(value, str) else value
                df = df[df['state'].isin(states)]
            elif key == 'institution_levels' and value is not None:
                if isinstance(value, (str, bytes)):
                    raise ValueError('institution_levels must contain codes 1, 2, or 3')
                levels = (
                    list(value)
                    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series))
                    else [value]
                )
                numeric_levels = pd.to_numeric(
                    pd.Series(levels), errors='coerce'
                )
                if (
                    numeric_levels.empty
                    or numeric_levels.isna().any()
                    or not np.isfinite(numeric_levels).all()
                    or not numeric_levels.isin([1, 2, 3]).all()
                ):
                    raise ValueError('institution_levels must contain codes 1, 2, or 3')
                institution_levels = pd.to_numeric(
                    df['institution_level'], errors='coerce'
                )
                df = df[institution_levels.isin(numeric_levels)]
            elif key in numeric_bounds and value is not None:
                column, bound = numeric_bounds[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value)
                ):
                    raise ValueError(f'{key} must be a finite number or null')
                metric_values = pd.to_numeric(df[column], errors='coerce')
                if bound == 'min':
                    df = df[metric_values >= value]
                else:
                    df = df[metric_values <= value]
            elif key in df.columns:
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

    df = df.copy()
    scaler = MinMaxScaler()
    df[['norm_price', 'norm_grad', 'norm_size']] = scaler.fit_transform(
        df[['net_price', 'graduation_rate', 'undergrad_size']]
    )

    w = np.array(user_weights)
    if not 0.0 <= ideal_size <= 1.0:
        raise ValueError('ideal_size must be between 0.0 and 1.0')
    ideal_vector = np.array([0.0, 1.0, ideal_size])
    user_vector = ideal_vector * w
    college_matrix = df[['norm_price', 'norm_grad', 'norm_size']].values * w

    similarities = cosine_similarity(college_matrix, user_vector.reshape(1, -1)).flatten()
    df['match_score'] = similarities * 100 # Convert to percentage

    prestige_features = pd.DataFrame(index=df.index)
    if 'admission_rate' in df:
        prestige_features['admission_rate'] = 1 - pd.to_numeric(
            df['admission_rate'], errors='coerce'
        )
    if 'average_sat_score' in df:
        prestige_features['average_sat_score'] = pd.to_numeric(
            df['average_sat_score'], errors='coerce'
        )
    if 'instructional_expenditure_per_student' in df:
        prestige_features['instructional_expenditure_per_student'] = pd.to_numeric(
            df['instructional_expenditure_per_student'], errors='coerce'
        )
    if 'carnegie_classification' in df:
        classification = pd.to_numeric(
            df['carnegie_classification'], errors='coerce'
        )
        prestige_features['carnegie_classification'] = (
            (classification == 15).astype(float) * 100
        ).where(classification.notna())

    prestige_scores = pd.DataFrame(index=df.index)
    for column in prestige_features:
        values = prestige_features[column]
        if column == 'carnegie_classification':
            prestige_scores[column] = values
        else:
            prestige_scores[column] = values.rank(pct=True) * 100
    df['prestige_score'] = prestige_scores.fillna(50).mean(axis=1).fillna(50)

    df['roi'] = np.nan
    df['roi_score'] = np.nan
    df['ranking_score'] = (
        (1.0 - prestige_weight) * df['match_score']
        + prestige_weight * df['prestige_score']
    )
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

    return df.sort_values(
        by=['ranking_score', 'match_score'], ascending=False
    ).head(10)