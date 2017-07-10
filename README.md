# EMNIST

Developed by @filipecasal

##### Description
This project is intended to train and explore some neural net models using the [EMNIST dataset](https://arxiv.org/abs/1702.05373 "EMNIST: an extension of MNIST to handwritten letters").

The project consists of 2 parts:

   Trainning scripts: baseline_nn.py, simple_cnn.py, larger_cnn.py and mnist_nn.py.
      
      Used to train the models.

   Predictor class: [predictor.py](https://github.com/filipecasal/emnist/blob/master/src/predictor.py)
      
      A class that can be started with the trained models and be used in other scripts.


##### Todo
   * Tune the models
   * Implement some unit tests
   * Implement some examples using the Predictor class


## Requirements

#### EMNIST Dataset (for the training)
  * Can be downloaded [here](http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip).

#### Python 3.5
##### requirements.txt
  * Keras==2.0.6
  * tensorflow==1.2.1
  * Theano==0.9.0
  * Pillow==4.2.1
  * h5py==2.7.0
  * boto==2.48.0
  * Flask==0.12.2

#### src/conf.py
  * Project Path
  * AWS Access key, Secret Key and Bucket name for model backup

#### Keras config
  * Configure Keras to use Theano backend
    $HOME/.keras/keras.json -> "backend": "theano"


## Usage

#### Training the neural nets (simple_cnn.py, larger_cnn.py, mnist_nn.py, baseline_nn.py)

A training script for each model. Models will be saved at models folder.

       usage:  *.py -f -m  [--epochs EPOCHS]

##### Arguments:

    -f FILE                 Path to the EMNIST FOLDER
    -m model_name           Model name to be saved
    [--epochs EPOCHS]       Epochs for the training


#### Show trained models accuracy

Calculate the accuracy of the models.

       usage:  example.py


#### Backup trained models to Amazon S3

A script to upload desired model to S3. (Need AWS config on src/conf.py)

      usage: src/import_to_s3.py -m

##### Arguments:

    -m model_name           Model name to be uploaded


#### Import trained models from Amazon S3

A script to upload desired model to S3. (Need AWS config on src/conf.py)

      usage: src/export_from_s3.py -m  

##### Arguments:

    -m model_name           Model name to be exported
