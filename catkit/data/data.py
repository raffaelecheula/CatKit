import catkit
import pathlib
import json

def get_properties():
    path = pathlib.Path(catkit.data.__file__).parents[0]
    with open(path / 'properties.json', 'r') as fileobj:
        properties = json.load(fileobj)
    return properties