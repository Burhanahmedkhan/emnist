import boto
import boto.s3
import argparse
from conf import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET
from conf import BASE_PROJECT_PATH


def connect():
    conn = boto.connect_s3(AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    return conn


def download_file(remote_model_name, local_model_path):
    print('Downloading {0}'.format(remote_model_name))
    conn = connect()
    bucket = boto.s3.bucket.Bucket(connection=conn, name=S3_BUCKET)
    key = bucket.get_key(remote_model_name)
    try:
        key.get_contents_to_filename(local_model_path)
    except:
        print("Modelo não existe no S3!")


def download_from_s3(model_name):

    model_yaml_remote = 'models/'+model_name+"/"+model_name+'.yaml'
    model_h5_remote = 'models/'+model_name+"/"+model_name+'.h5'
    mapping_model_remote = 'models/'+model_name+"/"+model_name+'_mapping.p'

    model_yaml_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.yaml'
    model_h5_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.h5'
    mapping_model_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'_mapping.p'

    download_file(model_yaml_remote, model_yaml_path)
    download_file(model_h5_remote, model_h5_path)
    download_file(mapping_model_remote, mapping_model_path)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Model name')
    args = parser.parse_args()
    download_from_s3(args.model)
