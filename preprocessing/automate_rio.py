import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def drop_outliers(df):
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include='number').columns
    Q1 = df_copy.quantile(0.25)
    Q3 = df_copy.quantile(0.75)
    IQR = Q3 - Q1
    df_copy = df_copy[~((df_copy[numeric_cols] < (Q1 - 1.5*IQR)) | (df_copy[numeric_cols] > (Q3 + 1.5*IQR))).any(axis=1)]
    return df_copy

def preprocessing_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = drop_outliers(df)
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other':2})
    df['smoking_history'] = df['smoking_history'].map({'No Info': 0, 'never': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5})

    X = df.drop(columns='diabetes')
    y = df['diabetes']

    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    return df_train, df_test

if __name__ == "__main__":
    df = pd.read_csv("diabetes_prediction_dataset.csv")

    df_train, df_test = preprocessing_data(df)

    df_train.to_csv("preprocessing/diabetes_prediction_preprocessing/diabetes_train.csv", index=False)
    df_test.to_csv("preprocessing/diabetes_prediction_preprocessing/diabetes_test.csv", index=False)