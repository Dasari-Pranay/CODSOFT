from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

def build_pipeline():
    categorical_features = ['Genre', 'Director', 'Actors']
    numeric_features = ['Duration', 'Year']

    # OneHotEncoder without 'sparse' argument for compatibility
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    return pipeline
