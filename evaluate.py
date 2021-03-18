import fire
import os
import pandas as pd
from evaluation import (
    compute_human_agreement,
    compute_rationale_consistency,
    get_testdata,
    get_annotations,
    ANNOTATION_DIR,
    ATTRIBUTION_DIR,
    DATA_FILE
)
from wandbinteraction import (
    runs_info
)


RESULTS_DIR = 'results'

def evaluate(fresh = False, metrics = ['HA', 'RA']):
    if 'HA' in metrics:
        df_annotations = get_annotations(ANNOTATION_DIR)
        HA_list = []
        for attribution_file in os.scandir(ATTRIBUTION_DIR):
            if attribution_file.name.endswith('.pkl'):
                HA = compute_human_agreement(df_annotations, attribution_file.name)
                HA_list.append(HA)
        HA_df = pd.DataFrame.from_records(HA_list)
        HA_df.to_csv(os.path.join(RESULTS_DIR, 'human_aggreement.csv'))

    if 'RC' in metrics:
        #df_info = pd.concat([runs_info(source, drop_large = True, token_types = tt) for source in ['scientsbank', 'beetle'] for tt in [True, False]])
        pass

    
if __name__ == '__main__':
    fire.Fire(evaluate)
