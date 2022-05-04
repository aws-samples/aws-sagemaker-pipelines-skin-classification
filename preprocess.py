"""Feature engineering the HAM10000 dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import sagemaker
import sagemaker.session

import numpy as np
import pandas as pd
import os
import cv2
        
import random
import shutil
from pathlib import Path
from PIL import Image,ImageOps,ImageEnhance

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import mxnet as mx
from tqdm import tqdm
from numpy.random import seed
seed(123)
import zipfile

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path) 
        
        
def fit(image, size, method=Image.BICUBIC, bleed=0.0, centering=(0.5, 0.5)):
    centering = list(centering)
    if not 0.0 <= centering[0] <= 1.0:
        centering[0] = 0.5
    if not 0.0 <= centering[1] <= 1.0:
        centering[1] = 0.5

    if not 0.0 <= bleed < 0.5:
        bleed = 0.0
    
    bleed_pixels = (bleed * image.size[0], bleed * image.size[1])

    live_size = (
        image.size[0] - bleed_pixels[0] * 2,
        image.size[1] - bleed_pixels[1] * 2,
    )

    # calculate the aspect ratio of the live_size
    live_size_ratio = live_size[0] / live_size[1]

    # calculate the aspect ratio of the output image
    output_ratio = size[0] / size[1]

    # figure out if the sides or top/bottom will be cropped off
    if live_size_ratio == output_ratio:
        # live_size is already the needed ratio
        crop_width = live_size[0]
        crop_height = live_size[1]
    elif live_size_ratio >= output_ratio:
        # live_size is wider than what's needed, crop the sides
        crop_width = output_ratio * live_size[1]
        crop_height = live_size[1]
    else:
        # live_size is taller than what's needed, crop the top and bottom
        crop_width = live_size[0]
        crop_height = live_size[0] / output_ratio

    # make the crop
    crop_left = bleed_pixels[0] + (live_size[0] - crop_width) * centering[0]
    crop_top = bleed_pixels[1] + (live_size[1] - crop_height) * centering[1]

    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    # resize the image and return it
    return image.resize(size, method, box=crop)
        
# custom function for image augmentation
def augment(img_in):
    flag_rotated = False
    if random.random() < 0.6:
        img_out = ImageOps.flip(img_in)
        
    if random.random() < 1.0:
        img_out = ImageOps.mirror(img_in)
    if random.random() < 0.9:
        img_out = img_in.rotate(180)
    if random.random() < 0.5:
        img_in = ImageOps.scale(img_in,factor=1.5)
        img_out = ImageOps.crop(img_in, border=16)
        
    if random.random() < 0.3 and not flag_rotated:
        flag_rotated = True
        img_in = img_in.rotate(30)
        left, top, right, bottom = 10,10,10,10
        img_in = img_in.crop((left, top, img_in.size[0] - right, img_in.size[1] - bottom))
        img_out= fit(img_in, img_size)
        
    if random.random() < 0.9 and not flag_rotated:
        flag_rotated = True
        img_in = img_in.rotate(-30)
        left, top, right, bottom = 10,10,10,10
        img_in= img_in.crop((left, top, img_in.size[0] - right, img_in.size[1] - bottom))
        img_out= fit(img_in, img_size)
    return img_out
 
# write images to recordio format    
def write_to_recordio(X: np.ndarray, y: np.ndarray, prefix: str):
    record = mx.recordio.MXIndexedRecordIO(idx_path=f"{prefix}.idx", uri=f"{prefix}.rec", flag="w")
    for idx, arr in enumerate(tqdm(X)):
        header = mx.recordio.IRHeader(0, y[idx], idx, 0)
        s = mx.recordio.pack_img(header,arr,quality=100,img_fmt=".jpg")
        record.write_idx(idx, s)
    record.close()

    
def identify_duplicates(x):
    unique_list = list(df['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
def identify_val_rows(x):
    ''' Creates a list of all the lesion_id's in the val set
    '''
    val_list = list(df_val['image_id'])
    
    if x in val_list:
        return 'val'
    else:
        return 'train'

    
if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    
    # upload HAM10000.zip to bucket and replace with your paths here
    skin_cancer_bucket='monai-bucket-skin-cancer' #replace this
    skin_cancer_bucket_path='skin_cancer_bucket_path' #replace this
    skin_cancer_files='dataverse_files' #replace this
    skin_cancer_files_ext='dataverse_files.zip' #replace this
    base_dir = "./" 

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()
    region = args.region
    prefix = args.prefix
    
    # set up the sagemaker session and the bucket where the outputs will be stored
    boto3.setup_default_session(region_name=region)
    boto_session = boto3.Session(region_name=region)
    s3 = boto3.client("s3", region_name=region)
    sagemaker_boto_client = boto_session.client("sagemaker")
    bucket = sagemaker.Session().default_bucket() 
    
    
    # check if directory exists
    if not os.path.isdir("data"):
        os.mkdir("data")

    # cleanup previous runs
    if os.path.exists(os.path.join(base_dir,skin_cancer_files)):
        shutil.rmtree(base_dir+skin_cancer_files)
    
    if os.path.exists(os.path.join(base_dir,skin_cancer_files_ext)):
        os.remove(os.path.join(base_dir,skin_cancer_files_ext))    

    data_dir = os.path.join(base_dir,'HAM10000')

    if os.path.exists(os.path.join(base_dir,'HAM10000.tar.gz')):
        os.remove(os.path.join(base_dir,'HAM10000.tar.gz'))

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    #Downloading training data set 
    os.mkdir(base_dir+skin_cancer_files)
    os.mkdir(base_dir+skin_cancer_files+'/HAM_images_part_1')
    os.mkdir(base_dir+skin_cancer_files+'/HAM_images_part_2')
    
    s3.download_file(skin_cancer_bucket, skin_cancer_bucket_path+'/'+skin_cancer_files_ext,base_dir+skin_cancer_files_ext)
    
    extract_archive(base_dir+skin_cancer_files_ext, base_dir+skin_cancer_files)
    extract_archive(base_dir+skin_cancer_files+'/HAM10000_images_part_1.zip', base_dir+skin_cancer_files+'/HAM_images_part_1')
    extract_archive(base_dir+skin_cancer_files+'/HAM10000_images_part_2.zip', base_dir+skin_cancer_files+'/HAM_images_part_2')
    
    # now we create 7 folders inside 'base_dir':
    os.mkdir(data_dir)
    
    # create a path to 'base_dir' to which we will join the names of the new folders
    train_dir = os.path.join(data_dir, 'train_dir')
    os.mkdir(train_dir)

    # val_dir
    val_dir = os.path.join(data_dir, 'val_dir')
    os.mkdir(val_dir)

    # Inside each folder we create seperate folders for each class
    # create new folders inside train_dir
    
    class_list = ['mel','bkl','bcc','akiec','vasc','nv','df']
    
    for class_name in class_list:
        class_name_folder = os.path.join(train_dir, class_name)
        os.mkdir(class_name_folder)

    # create new folders inside val_dir
    for class_name in class_list:
        class_name_folder = os.path.join(val_dir, class_name)
        os.mkdir(class_name_folder)
    
    df_data = pd.read_csv(base_dir+skin_cancer_files+'/HAM10000_metadata')
    #df_data = df_data.sample(frac=1).reset_index(drop=True)
    df = df_data.groupby('lesion_id').count()
    
    # now we filter out lesion_id's that have only one image associated with it
    df = df_data[df_data['image_id'] == 1]
    df.reset_index(inplace=True)

    # create a new colum that is a copy of the lesion_id column
    df_data['duplicates'] = df_data['lesion_id']
    
    # identify duplicates
    df_data['duplicates'] = df_data['duplicates'].apply(lambda x: 'no_duplicates' if df_data['lesion_id'].value_counts()[x] == 1 else 'has_duplicates')
    
    # now we filter out images that don't have duplicates
    df = df_data[df_data['duplicates'] == 'no_duplicates']
    y = df['dx']
    _, df_val = train_test_split(df, test_size=0.20, random_state=42, stratify=y)
    
    # create a new colum that is a copy of the image_id column
    df_data['train_or_val'] = df_data['image_id']
    
    # apply the function to this new column
    df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
   
    # filter out train rows
    df_train = df_data[df_data['train_or_val'] == 'train']
    df_data.set_index('image_id', inplace=True)
    
    #Get a list of images in each of the two folders
    folder_1 = os.listdir(base_dir+skin_cancer_files+'/HAM_images_part_1')
    folder_2 = os.listdir(base_dir+skin_cancer_files+'/HAM_images_part_2')

    # Get a list of train and val images
    train_list = list(df_train['image_id'])
    val_list = list(df_val['image_id'])
    
    # Transfer the train images
    for image in train_list:
        fname = str(image) + '.jpg'
        label = df_data.loc[image,'dx']
    
        if fname in folder_1:
            # source path to image
            src = os.path.join(base_dir+skin_cancer_files+'/HAM_images_part_1', fname)
            # destination path to image
            dst = os.path.join(train_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

        if fname in folder_2:
            # source path to image
            src = os.path.join(base_dir+skin_cancer_files+'/HAM_images_part_2', fname)
            # destination path to image
            dst = os.path.join(train_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        
    # define image size
    img_size = (224, 224)
    
    # Transfer the val images
    for image in val_list:
    
        fname = str(image) + '.jpg'
        label = df_data.loc[image,'dx']
    
        if fname in folder_1:
            # source path to image
            src = os.path.join(base_dir+skin_cancer_files+'/HAM_images_part_1', fname)
            # destination path to image
            dst = os.path.join(val_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

        if fname in folder_2:
            # source path to image
            src = os.path.join(base_dir+skin_cancer_files+'/HAM_images_part_2', fname)
            # destination path to image
            dst = os.path.join(val_dir, label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
            
    class_list = ['mel','bkl','bcc','akiec','vasc','df']
    img_idx = 0
    for item in class_list:    
        # We are creating temporary directories here because we delete these directories later
        # create a base dir
        aug_dir = data_dir + '/aug_dir'
        os.mkdir(aug_dir)
        # create a dir within the base dir to store images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        # Choose a class
        img_class = item

        # list all images in that directory
        img_list = os.listdir(os.path.join(train_dir, img_class))
       
        # Copy images from the class train dir to the img_dir e.g. class 'mel'
        for fname in img_list:
            # source path to image
            src = os.path.join(train_dir + '/' + img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir,fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
        
        # list all images in that directory
        aug_list = os.listdir(img_dir)
        
        #test: use build-in augmentation
        num_aug_images_wanted = 5000 # total number of images we want to have in each class
        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((num_aug_images_wanted/num_files)))
    
        j = 0
        for i in range(1, num_batches):
            for fname in aug_list:
                # source path to image
                src = os.path.join(img_dir, fname)
                im = Image.open(src)
                im_mirror = augment(im)
                dst = os.path.join(train_dir + '/' + img_class, 'AUG_' + str(j) + '_'+ fname)
                im_mirror.save(dst, quality=100)
            j = j + 1   
        
        shutil.rmtree(aug_dir)  
    
    # create file list and label list
    class_names = sorted([x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, x))])
    num_class = len(class_names)
    
    
    class_names = sorted([x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, x))])
    num_class = len(class_names)
    image_files = [[os.path.join(train_dir, class_name, x) 
                for x in os.listdir(os.path.join(train_dir, class_name))] 
               for class_name in class_names]
    
    image_label_list_train, images_list_train = [], []
    
    
    for i, class_name in enumerate(class_names):
        image_label_list_train.extend([i] * len(image_files[i]))
        for img in image_files[i]:
            image_temp = Image.open(img)
            image_temp = image_temp.resize(img_size, Image.ANTIALIAS)
            images_list_train.append(np.asarray(image_temp))
                                      
    print('Images by Class')
    print('nv: '+str(len(os.listdir(train_dir +'/nv'))))
    print('mel: '+str(len(os.listdir(train_dir +'/mel'))))
    print('bkl: '+str(len(os.listdir(train_dir +'/bkl'))))
    print('bcc: '+str(len(os.listdir(train_dir +'/bcc'))))
    print('akiec: '+str(len(os.listdir(train_dir +'/akiec'))))
    print('vasc: '+str(len(os.listdir(train_dir +'/vasc'))))
    print('df: '+str(len(os.listdir(train_dir +'/df'))))
    
    # since we were not augmenting the validation split, we will split it into test and validation dataset
    image_files = [[os.path.join(val_dir, class_name, x) 
                for x in os.listdir(os.path.join(val_dir, class_name))] 
               for class_name in class_names]

    images_list_val, images_list_test = [],[]
    image_label_list_val, image_label_list_test = [],[]
    
    for i, class_name in enumerate(class_names):
        # place first half of the validation in the validation split
        for j in range(0, int(len(image_files[i])/2)):
            image_temp = Image.open(image_files[i][j])
            image_temp = image_temp.resize(img_size, Image.ANTIALIAS)
            images_list_test.append(np.asarray(image_temp))
            image_label_list_test.append(i)
        # place second half of the validation in the test split    
        for j in range(int(len(image_files[i])/2), len(image_files[i])):
            image_temp = Image.open(image_files[i][j])
            image_temp = image_temp.resize(img_size, Image.ANTIALIAS)
            images_list_val.append(np.asarray(image_temp))
            image_label_list_val.append(i)
         
    print(len(image_label_list_train), len(image_label_list_val), len(image_label_list_test))
    
    write_to_recordio(images_list_val, image_label_list_val, prefix="data/val")
    write_to_recordio(images_list_test, image_label_list_test, prefix="data/test")
    write_to_recordio(images_list_train,image_label_list_train, prefix="data/train")
    
    # upload to the S3 bucket
    s3.upload_file("data/val.rec", bucket, f"{prefix}/data/val/val.rec")
    s3.upload_file("data/test.rec", bucket, f"{prefix}/data/test/test.rec")
    s3.upload_file("data/test.idx", bucket, f"{prefix}/data/test/test.idx")
    s3.upload_file("data/train.rec", bucket, f"{prefix}/data/train/train.rec")
    
