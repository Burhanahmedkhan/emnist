from flask import Flask, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
import re
import base64
from predictor import Predictor

app = Flask(__name__)


@app.route('/predict/', methods=['GET','POST'])
def predict():

    def parseImage(imgData):
        # parse canvas bytes and save as output.png
        imgstr = re.search(b'base64,(.*)', imgData).group(1)
        with open('output.png','wb') as output:
            output.write(base64.decodebytes(imgstr))

    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('images/output.png', mode='L')
    x = np.invert(x)
    x.shape
    # Visualize new array
    imsave('images/resized.png', x)
    x = imresize(x, (28, 28))

    # reshape image data for use in neural network
    # x = x.reshape(1,28,28,1)  # For the mnist_nn model
    # x = x.reshape(1, 784)  # For the baseline_nn model
    x = x.reshape(1, 1, 28, 28)  # For the simple_cnn model
    x.shape
    # Convert to float32
    x = x.astype('float32')

    # Normalize
    x /= 255

    response = predictor.predict(x)

    return response


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host URL')
    parser.add_argument('--port', type=int, default=5000, help='Host port')
    parser.add_argument('-m', '--model', default='model', type=str, help='Model name')
    args = parser.parse_args()

    predictor = Predictor(args.model)

    app.run(host=args.host, port=args.port, threaded=True)
