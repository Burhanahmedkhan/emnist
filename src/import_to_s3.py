import argparse
import boto
import boto.s3
from boto.s3.key import Key
from conf import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET
from conf import BASE_PROJECT_PATH


def connect():
    conn = boto.connect_s3(AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    return conn


def upload_file(local_file_to_upload, remote_path_to_upload):
    print('Uploading {0}'.format(local_file_to_upload))
    conn = connect()
    bucket = boto.s3.bucket.Bucket(connection=conn, name=S3_BUCKET)
    k = Key(bucket)
    k.key = remote_path_to_upload
    k.set_contents_from_filename(local_file_to_upload)


def upload_to_s3(model_name):

    model_yaml_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.yaml'
    model_h5_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'.h5'
    mapping_model_path = BASE_PROJECT_PATH+'/models/'+model_name+"/"+model_name+'_mapping.p'

    upload_file(model_yaml_path, 'models/'+model_name+"/"+model_name+'.yaml')
    upload_file(model_h5_path, 'models/'+model_name+"/"+model_name+'.h5')
    upload_file(mapping_model_path, 'models/'+model_name+"/"+model_name+'_mapping.p')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='model', type=str, help='Model name')
    args = parser.parse_args()
    upload_to_s3(args.model)
