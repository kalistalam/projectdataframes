import os
print("Current working directory:", os.getcwd())


import pandas as pd


def get_sub_group_type(df: pd.DataFrame) -> str:
    return str(df['Treatment_Group'].values[0])

def get_df_filtered_by_time_point(df_dict: pd.DataFrame, time_point: str) -> pd.DataFrame:
    
    sub_group_dict = dict()
    
    df: pd.DataFrame
    
    for PatientNumber, df in df_dict.items(): 
        
        # get sub group
        sub_group = get_sub_group_type(df)
        
        # filter by time point
        filtered = df[[time_point]].copy().astype(object)
        filtered.loc['sub_group', time_point] = sub_group
        filtered.rename(columns={time_point: PatientNumber}, inplace=True)
        filtered.name = PatientNumber
        
        sub_group_dict[PatientNumber] = filtered.T
        
    sub_group_df = pd.concat(sub_group_dict.values(), axis=0 )
    for col in sub_group_df.columns:
        if col != 'sub_group': 
            sub_group_df[col] = pd.to_numeric(sub_group_df[col], errors='coerce')
        else:
            sub_group_df[col] = sub_group_df[col].astype(str)
    return sub_group_df


def split_by_subgroup(df: pd.DataFrame) -> dict:

    subgroup_tables = {}

    for subgroup in df['sub_group'].unique():
        subgroup_df = df[df['sub_group'] == subgroup].copy()
        subgroup_df = subgroup_df.drop(columns=['sub_group'])
        subgroup_tables[subgroup] = subgroup_df

    return subgroup_tables

  
def main():

    all_baseline = {}
    all_week12 = {}
    
    df_dict = pd.read_excel('Patient_Hemodynamic_Data.xlsx', index_col='Assessment', sheet_name = None)
        
    baseline_df = get_df_filtered_by_time_point (df_dict, 'Baseline')
    print(baseline_df)
    
    week12_df = get_df_filtered_by_time_point (df_dict, 'Week12')
    print(week12_df)

    baseline_by_group = split_by_subgroup(baseline_df)
    week12_by_group = split_by_subgroup(week12_df)


    for group, df in baseline_by_group.items():
        all_baseline[group] = df.copy() 

    for group, df in week12_by_group.items():
        all_week12[group] = df.copy()
    
    new_placebo = (all_baseline['Placebo'] - all_week12['Placebo']).assign(sub_group='Placebo')
    new_20 = (all_baseline['Sildenafil 20mg TID'] - all_week12['Sildenafil 20mg TID']).assign(sub_group='20mg')
    new_40= (all_baseline['Sildenafil 40mg TID'] - all_week12['Sildenafil 40mg TID']).assign(sub_group='40mg')
    new_80= (all_baseline['Sildenafil 80mg TID'] - all_week12['Sildenafil 80mg TID']).assign(sub_group='80mg')

    whole_df = pd.concat([new_placebo, new_20, new_40, new_80], axis=0, ignore_index=False)

    return whole_df


from sklearn.linear_model import LogisticRegression

def run_logistic_regression(whole_df, threshold=5, target_col='Mean Pulmonary Arterial Pressure'):
    """
    Runs logistic regression on a DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and target column.
        threshold (float): Threshold for converting target to binary (default 5).
        target_col (str): Name of the numeric column to use as binary target.
    
    Returns:
        model: Trained LogisticRegression model.
        X (pd.DataFrame): Feature matrix used.
        y (pd.Series): Binary target vector.
        y_pred (np.ndarray): Predicted classes.
        y_prob (np.ndarray): Predicted probabilities.
        coeff_df (pd.DataFrame): DataFrame of feature coefficients.
    """
   
    whole_df= main()
    # Features = all columns except the target
    X = whole_df.drop(columns=[target_col])
    
    # Binary target
    y = (whole_df[target_col] > threshold).astype(int)
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    # Coefficients
    coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
    

    print("Predicted classes:\n", y_pred)
    print("\nPredicted probabilities:\n", y_prob)
    print("\nFeature coefficients:\n", coeff_df)

    return model, X, y, y_pred, y_prob, coeff_df

    

  
if __name__== "__main__":
    main()
    run_logistic_regression()   
