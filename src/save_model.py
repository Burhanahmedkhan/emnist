import pickle
import os
from keras.models import save_model
from src.conf import BASE_PROJECT_PATH


def save(model, mapping, model_name):
    os.makedirs(os.path.dirname(BASE_PROJECT_PATH+'/models/'+model_name+"/"), exist_ok=True)
    model_yaml_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.yaml'
    model_h5_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.h5'
    mapping_model_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'_mapping.p'



    model_yaml = model.to_yaml()
    with open(model_yaml_path, "w") as yaml_file:
        yaml_file.write(model_yaml)

    save_model(model, model_h5_path)

    pickle.dump(mapping, open(mapping_model_path, 'wb'))

    return
