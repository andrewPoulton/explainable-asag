import fire
import os
import pandas as pd
from evaluation import (
    evaluate_human_agreement
)


def evaluate():
    df =  evaluate_human_agreement('annotator/annotations', 'explained/scientsbank-1/31ktjuah.pkl')
    print(df)

if __name__ == '__main__':
    fire.Fire(evaluate)
