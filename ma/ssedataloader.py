
import pandas as pd

class SSEDataLoader:
    def __init__(self) -> None:
        self.path = "./datasets/"
    
    def load_data(self):
        filename = "stock_000799_20200102.csv"
        df = pd.read_csv(self.path+filename)
        df["tic"] = "000799"
        df["date"] = "20200102"
        return df

if __name__ == "__main__":
    loader = SSEDataLoader() 
    loader.load_data()
