import os
import pandas as pd

def process_college_data():
    # Ensure the Data directory exists locally
    os.makedirs('Data', exist_ok=True)
    
    print("Processing Institution-Level Dataset...")
    
    # 1. Define exact raw columns needed for the Institution table
    inst_cols = [
        'UNITID', 'INSTNM', 'STABBR', 'REGION', 'LOCALE', 
        'NPT4_PUB', 'NPT4_PRIV', 'C150_4', 'UGDS'
    ]
    
    # Load raw file (ensure your raw CSV is in the root or update path)
    inst_df = pd.read_csv('Data/Most-Recent-Cohorts-Institution.csv', usecols=inst_cols, low_memory=False)
    
    # Rename columns to clean, intuitive names
    inst_df = inst_df.rename(columns={
        'UNITID': 'unit_id',
        'INSTNM': 'school_name',
        'STABBR': 'state',
        'REGION': 'region_code',
        'LOCALE': 'locale_code',
        'NPT4_PUB': 'net_price_public',
        'NPT4_PRIV': 'net_price_private',
        'C150_4': 'graduation_rate',
        'UGDS': 'undergrad_size'
    })
    
    # Combine public and private net price into a single unified column
    inst_df['net_price'] = inst_df['net_price_public'].fillna(inst_df['net_price_private'])
    inst_df = inst_df.drop(columns=['net_price_public', 'net_price_private'])
    
    # Drop rows missing critical metrics to keep our vector math clean
    inst_df = inst_df.dropna(subset=['net_price', 'graduation_rate'])
    
    # Save to the Data folder
    inst_output_path = 'Data/institutions_clean.csv'
    inst_df.to_csv(inst_output_path, index=False)
    print(f" Saved clean institution table to {inst_output_path} ({len(inst_df)} schools)")


    print("\nProcessing Field-of-Study Dataset...")
    
    # 2. Define exact raw columns needed for the Program/Major table
    fos_cols = [
        'UNITID', 'INSTNM', 'CIPCODE', 'CIPDESC', 
        'CREDLEV', 'EARN_MDN_HI_1YR', 'DEBT_ALL_STGP_EVAL_MDN'
    ]
    
    fos_df = pd.read_csv('Data/Most-Recent-Cohorts-Field-of-Study.csv', usecols=fos_cols, low_memory=False)
    
    # Filter strictly for Bachelor's Degrees (CREDLEV == 3)
    fos_df = fos_df[fos_df['CREDLEV'] == 3]
    
    # Rename columns to human-readable names
    fos_df = fos_df.rename(columns={
        'UNITID': 'unit_id',
        'INSTNM': 'school_name',
        'CIPCODE': 'major_code',
        'CIPDESC': 'major_name',
        'CREDLEV': 'credential_level',
        'EARN_MDN_HI_1YR': 'median_earnings_1yr',
        'DEBT_ALL_STGP_EVAL_MDN': 'median_federal_debt'
    })
    
    # Drop unneeded credential level column now that we've filtered
    fos_df = fos_df.drop(columns=['credential_level'])
    
    # Clean up major names (strip whitespace, ensure proper formatting)
    fos_df['major_name'] = fos_df['major_name'].str.strip()
    
    # Save to the Data folder
    fos_output_path = 'Data/programs_clean.csv'
    fos_df.to_csv(fos_output_path, index=False)
    print(f" Saved clean program table to {fos_output_path} ({len(fos_df)} program records)")
    
    print("\n Preprocessing complete! Your optimized tables are ready in the Data/ folder.")

if __name__ == "__main__":
    process_college_data()