import pandas as pd
import json

def flatten_json(json_obj):
    """Función para aplanar un JSON anidado."""
    flat_dict = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            flat_dict[name[:-1]] = x

    flatten(json_obj)
    return flat_dict

def json_to_dataframe(json_file):
    # Leer el archivo JSON
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Verificar si el JSON es una lista o un objeto
    if isinstance(data, list):
        # Si es una lista de diccionarios
        # Aplanar cada diccionario y crear el DataFrame
        flattened_data = [flatten_json(item) for item in data]
        df = pd.DataFrame(flattened_data)
    else:
        # Si es un solo objeto JSON
        flattened_data = flatten_json(data)
        df = pd.DataFrame([flattened_data])
    
    return df

def getScoresFromJSON(filePath):
    df = json_to_dataframe(filePath)
    return df