from types import UnionType
from typing import Any
from sklearn.preprocessing import StandardScaler as SC
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split as tts
from copy import deepcopy

from data_container import PandasDataFrameContainer


class ColumnDropper:

    def __init__(self) -> None:
        pass

    def invoke(self, data:PandasDataFrameContainer) -> PandasDataFrameContainer:

        if data.is_splitted:
            data.X_train = data.X_train.drop(data.drop_cols, axis = 1)
            data.X_test = data.X_test.drop(data.drop_cols, axis=1)
        else:
            data.dataframe = data.dataframe.drop(data.drop_cols, axis = 1)

        return data

class StandardScaler:

    def __init__(self) -> None:
        self.scaler = SC()

    def invoke(self, data: PandasDataFrameContainer) -> PandasDataFrameContainer:
        
        if data.is_splitted:
            data.X_train[data.num_cols] = self.scaler.fit_transform(data.X_train[data.num_cols])
            data.X_test[data.num_cols] = self.scaler.transform(data.X_test[data.num_cols])
        else:
            data.dataframe[data.num_cols] = self.scaler.fit_transform(data.dataframe[data.num_cols])
        return data

class LabelEncoder():
    
    def __init__(self,) -> None:
        self.label_encoder_dict = None
        self.LabelEncoder = LE
    
    def invoke(self, data: PandasDataFrameContainer) -> PandasDataFrameContainer:

        if self.label_encoder_dict != None:
            print("Warning: Overwriting old label encoder data.")

        self.label_encoder_dict = dict.fromkeys(data.cat_cols, self.LabelEncoder())

        if (data.is_splitted):
            for col in data.cat_cols:
                data.X_train[col] = self.label_encoder_dict[col].fit_transform(data.X_train[col])
                data.X_test[col] = self.label_encoder_dict[col].transform(data.X_test[col])
        else:    
            for col in data.cat_cols:
                data.dataframe[col] = self.label_encoder_dict[col].fit_transform(data.dataframe[col])
            
        return data


class TrainTestSplit:

    def __init__(self) -> None:
        self.train_test_split = tts

    def invoke(self, data: PandasDataFrameContainer) -> PandasDataFrameContainer:

        # TODO add propper warnning here 
        if data.is_splitted:
            print("Warnning : data has already been splitted")

        data.X_train, data.X_test, data.y_train, data.y_test = self.train_test_split(data.dataframe, data.dataframe[data.target_col], test_size=data.test_size)
        data.is_splitted = True
        return data


class Chain():
    
    def __init__(self, func) -> None:
        self.func = func

    def __or__(self, func: Any) -> UnionType:
        
        class Step:

            def __init__(self, func) -> None:
                self.func = func

            def invoke(self, data: PandasDataFrameContainer):
                return func.invoke(self.func.invoke(data))
        
        return Chain(Step(self.func))
    
    def invoke(self, data:PandasDataFrameContainer) -> Any:
        return self.func.invoke(data)
