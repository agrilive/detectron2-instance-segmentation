from detectron2.data import DatasetCatalog, MetadataCatalog
import pickle

# load dataset dicts
with open('data/train.pkl', 'rb') as f:
    train_dicts = pickle.load(f)
with open('data/test.pkl', 'rb') as f:
    test_dicts = pickle.load(f)

# register dataset
d = 'train'
DatasetCatalog.register("pandasH_" + d, lambda d=d: train_dicts)
MetadataCatalog.get("pandasH_" + d).set(thing_classes=['panda', 'red_panda'])

d = 'test'
DatasetCatalog.register("pandasH_" + d, lambda d=d: test_dicts)
MetadataCatalog.get("pandasH_" + d).set(thing_classes=['panda', 'red_panda'])

print('Dataset registered')