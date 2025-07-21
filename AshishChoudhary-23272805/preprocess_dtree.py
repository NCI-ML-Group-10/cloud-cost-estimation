def clean_data(df):
    # Drop unnecessary columns
    return df.drop(columns=[
        'Resource ID', 'Usage Unit', 'Usage Start Date', 'Usage End Date',
        'Unrounded Cost ($)', 'Rounded Cost ($)'
    ])
