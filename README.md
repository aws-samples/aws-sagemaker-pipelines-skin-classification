## MLOps for existing workflow with SageMaker Pipelines: Skin lesion classification using Computer Vision

This repository aims at enabling the customization of a built-in SageMaker pipeline for MLOps to user-defined workflow. In this case we address a computer vision use case for skin lesion classification. This is a step-by-step guide on how to adapt an [existing code](https://github.com/aws-samples/amazon-sagemaker-monai-examples) to the CI/CD pipeline in AWS SageMaker Studio.


### <ins> Background </ins>

Amazon SageMaker is a fully-managed service for building, training an deploying Machine Learning (ML) models. It offers a variety of features. For instance SageMaker Pipelines is a continuous integration and continuous delivery (CI/CD) service designed for ML use cases. It can be used to create, automate, and manage end-to-end ML workflows. We are going to use the SageMaker Studio project template “**MLOps template for building, training, and deploying models**” that creates a pipeline and the infrastructure you need to create an MLOps solution for continuous integration and continuous deployment (CI/CD) of ML models. SageMaker Pipelines come with [SageMaker Python SDK](https//docs.aws.amazon.com/sagemaker/latest/dg/pipelines-build.html) integration, so you can build each step of your pipeline using a Python-based interface. SageMaker Studio Python SDK is an easy starting point, that allows Data Scientists to train and deploy models using popular deep learning frameworks with algorithms provided by Amazon, or their own algorithms built into SageMaker-compatible Docker images. For the detailed how-to create and access the template, please refer to [link](https://aws.amazon.com/blogs/machine-learning/building-automating-managing-and-scaling-ml-workflows-using-amazon-sagemaker-pipelines/).


### <ins> Implementation </ins>

#### I. AWS SageMaker Studio setup

1.  Open [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html). This can be done for existing users or while creating new ones. For a detailed how-to set up SageMaker Studio go [here](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html).
2. In [SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-launcher.html), you can choose the **Projects** menu on the [SageMaker resources](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-ui.html)** menu.

<img src="pictures/screenshot sm resources.png" width="300">


3. On the projects page, you can launch a pre-configured SageMaker MLOps template. Choose **MLOps template for model building, training, and deployment**.

![alt text](<pictures/screenshot create sm project.png>)

4. In the next page provide Project Name and short Description and select **Create Project**. The project will take a while to be created.


#### II. Prepare the dataset

1. Go to [Harvard DataVerse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
2. Select "**Access Dataset**" in top right, and review the license Creative Commons Attribution-NonCommercial 4.0 International Public License.
3. If you accept license, then select "**Original Format Zip**" and download the zip.
4. [Create S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) and choose a name starting with "sagemaker" (this will allow SageMaker to access the bucket without any extra permissions). You can enable access logigng and encryption for securtiy best practices. Upload dataverse_files.zip to it. Save the S3 bucket path for later use.
5. Make a note of the name of the bucket you have stored the data in, and the names of any subsequent folders, they will be needed later.


#### III. Preparing for data preprocessing

Since we will be using MXNet and OpenCV in our preprocessing step, we use a pre-built MXNet docker image and install the remaining dependencies using the requirements.txt file. To do so, you need to copy it and paste it under `pipelines/skin` in the **modelbuild** repository. 


#### IV. Changing the Pipelines template  

1. Create a folder inside the default bucket

2. Make sure the SageMaker Studio execution role has access to the default bucket as well as the bucket containing the dataset.

3. From the list of projects, choose the one that was just created.

5. On the **Repositories** tab, you can select the hyperlinks to locally clone the **CodeCommit** repositories to your local SageMaker Studio instance.

![alt text](<pictures/screenshot project created.png>)

5. Navigate to the **pipelines** directory inside the **modelbuild** directory and rename the **abalone** directory to **skin**.
6. Now open the **codebuild-buildspec.yml** file in the **modelbuild** directory and modify the run pipeline path from **run-pipeline —module-name pipelines.abalone.pipeline** (line 15) to this:

`run-pipeline --module-name pipelines.skin.pipeline \`

7. Save the file.
8. Replace 3 files in the Pipeline directory pipelines.py, preprocess.py and evaluate.py with the files from this repository.

<img src="pictures/screenshot folder structure.png" width="300">

9. Update the preprocess.py file (lines 183-186) with the S3 location (SKIN_CANCER_BUCKET) and folder name (SKIN_CANCER_BUCKET_PATH) where the dataverse_files.zip archive was uploaded to S3 at the end of the **Step II**:

* **skin_cancer_bucket**='monai-bucket-skin-cancer' (*replace this with your bucket name*)
* **skin_cancer_bucket_path**='skin_cancer_bucket_prefix' (*replace this with the prefix to the dataset inside the bucket*)
* **skin_cancer_files**='dataverse_files' (*replace this with name of the zip <ins>without</ins> extention*)
* **skin_cancer_files_ext**='dataverse_files.zip' (*replace this with name of the zip with extention*)

In the example above, the dataset would be stored under:
`s3://monai-bucket-skin-cancer/skin_cancer_bucket_prefix/dataverse_files.zip`

![alt text](<pictures/screenshot dataset file.png>)

10. Update line 127 in pipelines.py with URI of your docker created in **Step3.3**
`preprocessing_image_uri = <uri-to-your-ecr-container>`

#### V. Triggering a pipeline run  
Pushing committed changes to the CodeCommit repository (done on the Studio source control tab) triggers a new pipeline run, because an Amazon EventBridge event monitors for commits. We can monitor the run by choosing the pipeline inside the SageMaker project.  
![alt text](<pictures/screenshot commit and push.png>)

1.  To commit the changes, navigate to the Git Section on the left panel and follow the steps:
    a.  Stage all changes. You don't need to keep track of the -checkpoint file. You can add an entry to .gitignore file with `*checkpoint.*` to ignore them.
    b.  Commit the changes by providing a Summary and your Name and an email address.
    c.  Push the changes.

2. Navigate back to the project and select the **Pipelines** section.
3. If you double click on the executing pipelines, the steps of the pipeline will appear. You will be able to monitor the step that is currently running.

![alt text](<pictures/screenshot pipeline execution.png>)

4.  When the pipeline is complete, you can go back to the project screen and choose the **Model groups** tab. You can then inspect the metadata attached to the model artifacts.

5.  If everything looks good, you can click on the Update Status tab and manually approve the model. The default ModelApprovalStatus is set to PendingManualApproval. If our model has greater than 60% accuracy, it’s added to the model registry, but not deployed until manual approval is complete. You can then go to **Endpoints** in the SageMaker menu where you will see a staging endpoint being created. After a while the endpoint will be listed with the **InService** status.

6. To deploy the endpoint into production, go to CodePipeline, click on the **modeldeploy** pipeline which is currently in progress. At the end of the DeployStaging phase, you need to manually approve the deployment. Once it is done you will see the production endpoint being deployed in the SageMaker Endpoints. After a while the endpoint will also be **InService**.

![alt text](<pictures/screenshot DeployStaging step.png>)


### <ins> Dataset </ins>

The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions: [Web Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)


### <ins> Useful links </ins>

* https://github.com/aws/sagemaker-python-sdk/blob/dev/src/sagemaker/amazon/README.rst gives an example with PCA and a link to all SM algorithms
*  https://sagemaker.readthedocs.io/en/stable/frameworks/index.html gives a list of frameworks (estimators) supported by SM with links to examples
* https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html list of Amazon SageMaker Built-in Algorithms

*  https://docs.docker.com/engine/reference/builder/ "The official Dockerfile reference guide"

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
