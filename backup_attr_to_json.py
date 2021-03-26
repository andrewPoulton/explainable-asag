import os
from evaluation import AttributionData
for dirpath, dirs, files in os.walk("attributions"):
    for d in dirs:
        os.makedirs(os.path.join('attributions_json', d))
        for f in os.listdir(dirpath, d):
            if f.endswith('.pkl'):
                a = AttributionData(os.path.join(dirpath, d, f))
                a.to_json(os.path.join('attributions_json', d, os.path.splitext(f)[0]+'.json'))
))
