def extract_features(df):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd

    cat_cols = ['Service Name', 'Region/Zone']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]),
                           columns=encoder.get_feature_names_out())

    df_num = df.drop(columns=cat_cols).reset_index(drop=True)
    final_df = pd.concat([df_num, encoded.reset_index(drop=True)], axis=1)
    return final_df
