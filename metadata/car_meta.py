import os
import json

# Get all the paths in the directory
dir_path = '/share1/jialuo/car/02958343/'
paths = [os.path.join(dir_path, dirname + '/models/model_normalized.obj') for dirname in os.listdir(dir_path)]

paths = [path for path in paths if os.path.exists(path)]
print(len(paths))

# Save the paths to a JSON file
with open('cars_sdf.json', 'w') as f:
    json.dump(paths, f)