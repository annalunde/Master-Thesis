import pandas as pd
import os
from decouple import config
import numpy as np


class AnalyserDisruptions:
    def __init__(self, data_path):
        self.data_path = data_path

    # NO SHOW
    # analyse av antall no shows per totalt antall requests
    # average antall no shows per måned
    # average tid i mellom hver no show
    # distribusjon ventetid sjåfør
    def no_show(self):
        df = pd.read_csv(self.data_path)


    # DELAY
    # hva defineres som en delay? legge til at man kan endre hvor mange minutter som defineres som en delay
    # analyse av antall delays per totalt antall requests
    # average antall delays per måned
    # average tid i mellom hver delay
    # distrubisjon av lengde av hver delay

    # CANCEL
    # kanselleringen må skje samme dag som den originale requesten for å bli tatt med - teller kun på systemet da
    # analyse av antall cancels per totalt antall requests
    # average antall cancels per måned
    # average tid i mellom hver cancel
    # distrubisjon av lengde av hver cancel

def main():
    analyser = None

    try:
        analyser = AnalyserDisruptions(
            data_path=config("data_processed_path"))

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
