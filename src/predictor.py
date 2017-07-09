import numpy as np
from keras.models import model_from_yaml
import pickle
from src.conf import BASE_PROJECT_PATH


class Predictor():

    def __init__(self, model_name):
        # load YAML and create model
        with open(BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.yaml', 'r') as yaml_file:
            loaded_model_yaml = yaml_file.read()
        self.model = model_from_yaml(loaded_model_yaml)

        # load weights
        self.model.load_weights(BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.h5')
        self.mapping = pickle.load(open(BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'_mapping.p', 'rb'))

    def predict(self, x):
        # reshape image data for use in neural network
        # x = x.reshape(1,28,28,1) ## For the mnist_nn model
        # x = x.reshape(1, 784) ## For the baseline_nn model
        # x = x.reshape(1, 1, 28, 28)  # For the simple_cnn model

        # Convert to float32
        # x = x.astype('float32')

        # Normalize
        # x /= 255

        # Predict
        out = self.model.predict(x)

        # Generate response
        response = chr(self.mapping[(int(np.argmax(out, axis=1)[0]))])
        return response
