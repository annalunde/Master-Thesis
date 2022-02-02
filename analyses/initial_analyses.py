import pandas as pd
from decouple import config


class Analyser:
    def __init__(self, data_path):
        self.data_path = data_path

    def return_probability(self):
        df = pd.read_csv(self.data_path)
        print(df.head())
        dups = df.pivot_table(columns=['Request ID'], aggfunc='size')
        print(2 in dups)

    # probability of return out of total requests

    # per time segment, how possible is it that there is a return?


def main():
    analyser = None

    try:
        analyser = Analyser(
            data_path=config("data_processed_path"))
        print("Analysing return probability distributions")
        analyser.return_probability()

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
