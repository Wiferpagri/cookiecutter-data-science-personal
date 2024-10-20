import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_type: str = 'standard',
    encoder_type: dict = None
) -> pd.DataFrame:
    """
    Preprocesses the train and test DataFrames by scaling numerical features and encoding categorical features.
    The function allows for dynamic selection of scalers for numerical features and encoders for categorical features.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        The training dataset containing both numerical and categorical features.
    
    test_df : pd.DataFrame
        The testing dataset containing both numerical and categorical features.
    
    scaler_type : str, optional (default='standard')
        The type of scaler to apply to numerical features. Supported options are:
        - 'standard': StandardScaler (default) - standardizes features by removing the mean and scaling to unit variance.
        - 'minmax': MinMaxScaler - scales features to a range between 0 and 1.
        - 'robust': RobustScaler - scales features using the median and interquartile range, less sensitive to outliers.
        - 'power': PowerTransformer - applies a power transformation to make data more Gaussian-like.
    
    encoder_type : dict, optional (default=None)
        A dictionary specifying the encoding method for each categorical column. If None, all categorical features are 
        encoded using OneHotEncoder by default. The supported encodings for categorical columns are:
        - 'label': LabelEncoder - encodes each label with a unique integer (applies to a single column).
        - 'ordinal': OrdinalEncoder - encodes categories with ordinal values (in ascending order).
        - 'onehot': OneHotEncoder - encodes categorical features as one-hot arrays (used by default).
        Example: 
        encoder_type = {
            'col_categorical_1': 'onehot', 
            'col_categorical_2': 'ordinal', 
            'col_categorical_3': 'label'
        }

    Returns:
    --------
    train_df_preprocessed : pd.DataFrame
        The preprocessed training dataset with scaled numerical features and encoded categorical features.

    test_df_preprocessed : pd.DataFrame
        The preprocessed testing dataset with scaled numerical features and encoded categorical features.
    
    Notes:
    ------
    - Numerical features are scaled according to the specified `scaler_type`.
    - Categorical features are encoded according to the specified `encoder_type` for each column.
    - If no specific encoding is provided for a categorical column, OneHotEncoder with drop='first' is used by default.
    - The function converts the transformed arrays back into DataFrames with appropriate column names.

    Example usage:
    --------------
    encoder_dict = {
        'col_1': 'onehot',
        'col_2': 'ordinal',
        'col_3': 'label'
    }
    train_df_preprocessed, test_df_preprocessed = preprocessing(train_df, test_df, scaler_type='minmax', encoder_type=encoder_dict)
    """
    num_feat = train_df.select_dtypes(include=['number']).columns
    cat_feat = train_df.select_dtypes(include=['object', 'category']).columns
    
    # Selecting scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler == 'power':
        scaler = PowerTransformer() # Normal-like transform
    else:
        scaler = StandardScaler() # Standardization by default
    
    transformers = [('num', scaler, num_feat)]
    
    # Encoding categorical columns
    for cat_col in cat_feat:
        # If the user chosen different encoding for each column
        if encoder_type and cat_col in encoder_type:
            if encoder_type[cat_col] == 'label':
                encoder = LabelEncoder()
            elif encoder_type[cat_col] == 'ordinal':
                encoder = OrdinalEncoder()
            else:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
        else:
            encoder = OneHotEncoder(sparse_output=False, drop='first')
    
        # Adding transformer of the categorical column
        transformers.append(f'cat_{cat_col}', encoder, [cat_col])
    
    # Preprocessing
    preprocessor = ColumnTransformer(transformers=transformers)
    
    train_preprocessed = preprocessor.fit_transform(train_df)
    test_preprocessed = preprocessor.transform(test_df)
    
    # Converting them to DataFrames
    # Obtaining numerical column names
    num_col_names = num_feat.tolist()
    
    # Obtaining categorical column names
    cat_col_names = []
    for cat_col in cat_feat:
        if encoder_type and encoder_type.get(cat_col) == 'onehot':
            cat_col_names.extend(preprocessor.named_transformers_[f'cat_{cat_col}'].get_feature_names_out([cat_col]).tolist())
        else:
            cat_col_names.append(cat_col)
        
    all_col_names = num_col_names + cat_col_names
    
    train_df_preprocessed = pd.DataFrame(train_preprocessed, columns=all_col_names)
    test_df_preprocessed = pd.DataFrame(test_preprocessed, columns=all_col_names)
    
    return train_df_preprocessed, test_df_preprocessed
