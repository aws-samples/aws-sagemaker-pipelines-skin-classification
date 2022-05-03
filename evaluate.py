"""Evaluation script for measuring mean squared error."""
import json
import logging
import argparse
import pathlib
import pickle
import tarfile
import boto3
import sagemaker
import os
import numpy as np
import pandas as pd
import mxnet as mx
import time
import PIL.Image as Image
import io

from sagemaker import image_uris, session
from sagemaker.model import Model
from sagemaker.predictor import RealTimePredictor
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix,classification_report

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def softmax(x):
    a = np.exp(x)/np.sum(np.exp(x))
    return a


if __name__ == "__main__":
    # parse arguments
    logger.debug("Starting evaluation.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--modelartifact", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    
    args = parser.parse_args()
    region = args.region
    role = args.role
    prefix = args.prefix
    model_path = args.modelartifact
    
    boto3.setup_default_session(region_name=region)
    client = boto3.client('runtime.sagemaker')
    bucket = sagemaker.Session().default_bucket()
    s3 = boto3.client("s3", region_name=region)

    # load model using the same image classification image or another image of your choice
    logger.debug("Loading model.")
    training_image = sagemaker.image_uris.retrieve("image-classification", region)
    model = Model(model_data=model_path, 
                  image_uri=training_image,
                  role=role,
                  predictor_cls=RealTimePredictor)

    # read data from S3 bucket: download data and read image and label from record
    logger.debug("Reading test data.")
    
    if not os.path.exists('data'):
        os.mkdir('data')
        
    precision = recall = f1 = 0
    for data_type in ["test"]: 
        # download test files
        s3.download_file(bucket, f"{prefix}/data/{data_type}/{data_type}.rec", f"data/{data_type}.rec")
        s3.download_file(bucket, f"{prefix}/data/{data_type}/{data_type}.idx", f"data/{data_type}.idx")

        record = mx.recordio.MXIndexedRecordIO(idx_path=f"data/{data_type}.idx", uri=f"data/{data_type}.rec", flag="r")

        X_test, y_test = [], []
        for i in range(len(record.keys)): 
            try:
                item = record.read_idx(i)
                header, s = mx.recordio.unpack_img(item)
                X_test.append(s)
                y_test.append(int(header.label))
            except:
                print(f"Error with loading item {i}")
                break
        record.close()

        # run inference on test data
        logger.info(f"Performing predictions against {data_type} data.")

        timestamp = time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())
        endpoint_name = "skin-classfication-" + timestamp
        predictor = model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge', role=role, endpoint_name=endpoint_name)

        predictions = []
        for img in X_test:
            pil_im = Image.fromarray((np.array(img)))
            b = io.BytesIO()
            pil_im.save(b, 'jpeg')
            im_bytes = b.getvalue()
    
            response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-image', Body=im_bytes)
            np_bytes = response['Body'].read()
            array_probs = np.asarray(np_bytes.decode("utf-8").replace("[","").replace("]","").split(","),dtype=np.float32)
            predictions.append(np.argmax(array_probs,axis=0))


        logger.debug("Calculating precision, recall and F1 score.")

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
        logger.debug(precision, recall, f1)
        
    # generate evaluation report 
    report_dict = {
        "classification_metrics": {
            "precision": {
                "value": precision,
            },
            "recall": {
                "value": recall,
            },
            "f1_score": {
                "value": f1,
            },
        },
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with precision: %f, recall: %f and F1 score: %f", precision, recall, f1)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    # clean up
    predictor.delete_endpoint()
