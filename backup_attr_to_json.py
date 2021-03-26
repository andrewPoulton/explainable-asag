import os
from evaluation import AttributionData
print('Copying attribution pickle to json.')
for dirpath, dirs, files in os.walk("attributions"):
    for d in dirs:
        os.makedirs(os.path.join('attributions_json', d), exist_ok= True)
        for f in os.listdir(os.path.join(dirpath, d)):
            if f.endswith('.pkl'):
                pickle_file = os.path.join(dirpath, d, f)
                json_file = os.path.join('attributions_json', d, os.path.splitext(f)[0]+'.json')
                print('pickle:', pickle_file)
                print('json:', json_file)
                a = AttributionData(pickle_file)
                a.to_json(json_file)
