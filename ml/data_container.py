from pandas import DataFrame
from typing import List

class PandasDataFrameContainer:
    """
        Class for storing data and metadata while it is getting processed.
    """

    def __init__(   self, 
                    dataframe   : DataFrame,
                    num_cols    : List[str],
                    cat_cols    : List[str],
                    drop_cols   : List[str],
                    target_col  : str,
                    test_size   : float     =   0.3) -> None:
        """
            Class Constructor

            param:
                dataframe   (pandas dataframe)  : whole dataset
                num_cols    (list[str])         : names of numberical columns
                cat_cols    (list[str])         : names of categorical columns
                drop_cols   (list[str])         : columns to be dropped
                target_col  (str)               : name of target column
                test_size   (float)             : percentage of test size in float ranging from [0.0 to 1.0]
        """

        self.dataframe = dataframe

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.drop_cols = drop_cols
        self.target_col = target_col

        self.test_size = test_size
        
        # variable to signify that the dataset has been splitted or not
        self.is_splitted = False

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def valide(self,):
        """
            Function to validte the data. Raises an error if there is a missmatch in the data.

            param: None
            return: None
        """

        # TODO raise proper errors if validation fails
        if(self.is_splitted and None in [self.X_train, self.X_test, self.y_train, self.y_test]):
            print("split error")