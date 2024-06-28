# Challenge Documentation

This documentation describes the different stages of the development of the challenge and key decisions that were made. It ends with some closing remarks and enhancements that should be done to the solution.

## Setup

Although it is not solicited as part of the solution, some initial configurations were made to ensure good quality practices during the development of the challenge. Code formatting and style checking tools `black`, `flake8` and `isort` were installed, and a settings file for VSCode was created (located at `.vscode/settings.json`) so that the tools run automatically on file save and check the code for any formatting issues. This is a quick setup that ensures good formatting and easier-to-read code.

## Notebook reproduction

The first step on the migration of the explored model was to reproduce the notebook created by the Data Scientist (DS from now on).

This notebook had some bugs that were fixed along the way to be able to run it correctly or avoid getting misleading results. The found bugs were:

* Positional arguments on the calls to `seaborn` plotting functions were producing an error. The bug was fixed by passing the name of the arguments (`x` and `y`) instead of positional arguments.
* The feature generation function `is_high_season` had problems with the upper bounds of the ranges. These upper bounds had a time set of "00:00:00" (beggining of the day), so the flights with date equal to the upper bound didn't get transformed correctly. The bug was fixed by removing the time information from the `Fecha-I` string and just keeping the date, so that the comparison is done only between dates.
* The function `get_rate_from_column` was computing `# total flights / # delayed flights` instead of the actual delay rate which is `# delayed flights / # total flights`. Fixing this gives better insights of the data.
* The first XGBoost model trained (with all the features and no balance) predicted all 0s on the test set, which indicates that the model didn't train correctly. The simplest reason may be that the `learning_rate` is too small and doesn't allow the model to learn much. Adjusting the learning rate to be a little bit higher (~0.5) helps with the training in this case. However, so that the top 10 features selected by DS don't change, we will keep the `learning_rate` as is.
* The selected top 10 features are not even the top 10 features with highest feature importance, and there's no argument to why they were chosen. This is left as is to avoid affecting the output model, although better feature selection should be done on the future.

Also, there were some missing required missing dependencies to reproduce the notebook (`ipykernel` and `xgboost`). These were added to the `requirements-dev.txt` requirements file.

### Model choice

At the end of the notebook, the DS had trained six different models:
* XGBoost/Logistic Regression model with all the features and no class weights
* XGBoost/Logistic Regression model with top 10 features and no class weights
* XGBoost/Logistic Regression model with top 10 features and class weights

As the DS correctly states, all the models trained with no class weights performed poorly on the `Delayed` class (1), mainly in terms of recall.

Ideally, there should be a discussion with the stakeholders about what metrics to optimize for the `Delayed` class: do they prefer high precision? or high recall? Since this information is not available, it's safe to assume that both metrics have equal importance, and that the F1-score of the `Delayed` class should be optimized, without significantly affecting the metrics for the `Not-Delayed` (0) class.

Both models trained with top 10 features and class weights achieved virtually the same metrics: an F1-score of ~0.36 on the `Delayed` class and ~0.65 on the `Not-Delayed` class, although results were slightly better for the XGBoost model.

Since both models achieved equal metrics, the inference time was analyzed (see the Annex section on the `exploration.ipynb` notebook) to choose the one that's faster. In this section, it can be seen that the Logistic Regression model tends to have better and more stable inference time. This makes sense, as this model tends to be more lightweight than an XGBoost model, and should also provide better training times. Since Logistic Regression is conceptually more simple, it is also more interpretable than XGBoost, which makes it the preferred candidate to push to production.

To summarize, Logistic Regression is chosen since it achieves the same performance metrics and is faster and more simple.

## Model migration

The idea of this step was to migrate the model from the exploration notebook to a fully functional Python script for preprocessing the data, training the model and providing inference. The methods of the class `DelayModel` are completed to achieve this. Here are some special considerations and observations from this step:

* The `pd.get_dummies` method was not used to perform One-Hot Encoding. Instead, custom code was built to do it. The reason is that, if one of the categories of the features to One-Hot Encode is not present on the data, `get_dummies` will not create a column for that category, whereas the correct thing would be to create the column for that category with all 0s. With this custom code, only the categories selected by the DS get encoded and if any of them is not present on the data, it gets filled with 0s.

* There are some issues with the test script `test_model.py`. First, the path to the data is incorrect and needed to be changed in order to run the tests. Then, the `test_model_predict` test initializes a `DelayModel` object and tries to run the `predict` method without never calling the `fit` method first. This is unacceptable and, ideally, an exception should be raised indicating that the model has not been trained. However, the test expects the `predict` method to return a list of ints, independent of this error, which is considered undesirable. Since the test file can't be changed, the choice was to return a dummy list filled with a very large negative value (`-2**60`) so that the test runs correctly but results make no sense, since no trained model is being used.

## API development

On this step, the goal was to serve the trained model and provide an endpoint for inference.

The API should load the model on startup from a local file. In order to do this, the `model.py` script was extended with a `save` method which writes the `DelayModel` object to a `pickle` file on a given path. Then, a model was trained over all the available data and stored locally with the following code:

```
from model import DelayModel
import pandas as pd

# Read the data
df = pd.read_csv("data/data.csv")

# Create the model
model = DelayModel()

# Preprocess the data
X_train, y_train = model.preprocess(df, "delay")

# Train the model
model.fit(X_train, y_train)

# Store the model
model.save("challenge/tmp/model_checkpoint.pkl")
```

The API reads this `pickle` file on startup for loading the trained model.

Here are some important details about this step:

* Due to the way the API tests are built, the API startup methods don't get invoked and the API doesn't get initialized if a `with` statements is not used ([reference](https://github.com/tiangolo/fastapi/issues/1072)). Since the test file can't be changed, it's not possible to use FastAPI's startup event or `lifespan` method for initializing the API, which is the recommended way of loading the model on startup. To circumvent this, the model is directly initialized on the `api.py` script, which is undesirable since the model will be loaded even if the `api.py` script or a part of it is only being imported.

* `pydantic` was used to define the input schema and the input validations. A custom exception handler needed to be built for the `RequestValidationError` so that a status code 400 is returned instead of the default 422 "Unprocessable Entity". The change was made to fit the tests provided on the `test-api` suite.


## API deployment

The goal of this step was to deploy the API on a public endpoint. To complete it, a Docker image that runs the server with the API is built and then deployed to GCP.


### Docker image

The provided Dockerfile was completed so that the image is built and the API is started when the image runs inside a container.

In order to use a Python version similar to the one used throughout the development, the base image was changed to be the `3.9-slim`. This is a more lightweight base image and it is closer to the version used during development (`3.9.4`).

### GCP

The API was deployed as a Cloud Run Service that exposes a public endpoint. In order to accomplish this, these were the steps taken:

1. A GCP project was created with the ID `rodrigo-tryolabs-latam`.
2. The Artifact Registry is used to store the Docker image of the API. A repository for the image needs to be created with the following command:

    `gcloud artifacts repositories create --repository-format=docker --location=us-west1 delay-model-service`

    This creates a repository on the address `us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service`.

3. The next step is to build the Docker image and tag it with the address of the created repository:

    `docker build . -t us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service/delay-model-api`

4. Before pushing the image to the remote Artifact Registry repository, Docker needs to be authenticated to GCP. This can be done with the command:

    `gcloud auth configure-docker us-west1-docker.pkg.dev`

5. Next, the image is pushed to the repository:

    `docker push us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service/delay-model-api`

6. Finally, image is deployed as a Cloud Run Service with the command:

    ```
    gcloud run deploy delay-model \
        --image us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service/delay-model-api \
        --allow-unauthenticated \
        --region us-west1
    ```

    This command deploys the previously uploaded Docker image as a service. The argument `--allow-unauthenticated` publicly exposes the API and allows for unauthenticated requests. Note that some default arguments are used: the CPU limit is set to 1 vCPU; the memory limit is set to 512MiB; the service is shutdown when idle.

After the deployment is completed, the API is available at https://delay-model-dpmrk4cwxq-uw.a.run.app, and the prediction endpoint is available at https://delay-model-dpmrk4cwxq-uw.a.run.app/predict. **This is not the final endpoint to evaluate the solution**. Instead, this endpoint was deployed for testing purposes, using Postman or the provided stress test suite.

The results of the stress test on this endpoint were an error rate of 0%, an average response time of 343ms and a maximum response time of 743ms. Also, the API was able to respond to 87.69 requests per second.


## CI/CD Pipeline

On this final step, the goal was to setup a proper CI/CD pipeline.

The Continuous Integration (CI) workflow focuses on running the tests and assesing the quality of the code each time there's a push to the repository, with the goal of detecting bugs earlier, correcting code faster and ensuring good code quality practices.

The Continuous Deployment (CD) workflow focuses on training the model, deploying the API and running the stress test against it. This workflow only runs when there's a push to the `main`, `develop` or `release` branches.

### Continuous Integration

The goals of this workflow are checking the code quality and testing it. For the first goal, the code is checked using `black`, `flake8` and `isort` to ensure that the style and format are correct and fit the repository standards. For the second goal, the provided test suites (`model-test` and `api-test`) are ran to ensure that the changes done on the code don't affect the functionality of the `DelayModel` class and the API.

Important decisions made on this step:

* The test suites require a trained model available for testing purposes. However, these test suites run on Github workers and don't have access to local models. To circumvent this, the model checkpoint is tracked with Git and uploaded to the remote. This is not desirable, since models can grow rapidly in size and managing them inside the repository can become a problem. The ideal solution would be to maintain a proper Model Registry, with remote storage and a good version management, so that trained models can be uploaded to it or downloaded for testing or deployment. Due to time restrictions and since this particular model checkpoint is lightweight, the decision to track the model was taken.
* The `model-test` suite had to be modified due to an error. The path to the data file on the suite was `../../data/data.csv`, which assumed that the tests were ran from the `tests/model/` directory, but tests should actually be ran from the project root folder, where the `Makefile` is. To fix this, the path was changed to `data/data.csv`. With this change, tests run correctly and can be triggered from the GA.

### Continuous Deployment

The goal of this workflow is to train the model, build the Docker image with it and deploy it to a Cloud Run service. This workflow only runs when there's a push to the `main`, `develop` or `release` branches and it deploys a different API for each of these. The reasoning is that having different deployments for different stages of the development of features and releases can help in testing how the changes affect the deployment, while keeping the `main` API intact and serving only the released code features.

Here are the most important steps taken to develop this workflow:

* A small and simple training script (`train.py`) was created so that the GA trains the model before deploying it. This training script uses all the available data, preprocesses it, trains the model and writes it to the location used by the Dockerfile to put the model inside the Docker image. This is a simplification of a real scenario. Ideally, the data would be stored remotely and we would have different remote jobs for preprocessing the data, training the model and uploading it to a Model Registry. These remote jobs could be triggered by the same events that trigger this workflow, but none of the preprocessing or training would run inside the GA synchronously.
* A GCP Service Account `cd-pipeline-sa` was created to grant the Github Action runner with permissions to push the Docker image to the Artifact Registry repository and to deploy the Cloud Run Service. The roles given to this SA are:
    - `Artifact Registry Writer`: enables the SA to push Docker images to the Artifact Registry repositories.
    - `Cloud Run Admin`: gives the SA full control over the Cloud Run services deployed.
    - `Service Account User`: gives the SA the necessary permissions to act as the default Cloud Run service account. This permission is needed for deploying from the Github Action.
We created one single SA for simplification, since we only use it in a single workflow. Ideally, we should have multiple SAs, each with more granular and reduced permissions; for example, we could have a "Cloud Run SA" which only has control over the services and nothing else, and a separate "Artifact Registry SA" which only has access to the repository.
* A `dev` environment was created on the Github Repository, containing various configuration variables (mostly names used through the GCP deployment) and secrets (the key to access the SA `cd-pipeline-sa`). The created configuration variables are:
    - `GAR_IMAGE_NAME=delay-model-api`
    - `GAR_REPOSITORY=us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service`
    - `GCP_PROJECT_ID=rodrigo-tryolabs-latam`
    - `GCP_REGION=us-west1`
    - `GCR_SERVICE_NAME=delay-model`
* After deployment of the service, the stress tests run against the deployed API. As mentioned, different APIs are deployed depending on the branch. To point the stress test script to the correct API, a small modification was needed to be done to the `Makefile`, so that the URL of the API is passed as an argument on the `make stress-test` command. The final command is `make stress-test API_URL=<api-url>`.

## Closing Thoughts and Enhancements

All the steps of the challenge were completed. Each step had it's individual PR, with a description of the changes to complement the sections documented on this file. To finalize,  a `release` branch is carved out of the `develop` branch, and after testing the API deployed for the `release` branch and assesing that the GAs run correctly, the `release` branch is merged into the `main` branch, with a `v1.0.0` tag created on the merge commit. All of these changes and merges can be reviewed inside the repository. All of the feature branches and PRs are left there for review. The provided testing endpoint is the URL of the service deployed for the `main` branch.

Although the provided solutions accomplishes the goals of the challenge, several decissions were made which are not optimal and should be improved on a real production scenario. These decisions were made due to time restrictions or to fit the limitations of the challenge and the test suites. Various improvements were discussed throughout the documentation, but here we highlight some of the most important. A priority level ('nice to have' or 'must have') is also included assesing how vital this improvement is for a real solution.

### Multiple deployment environments

All of the developments and deployments were done on the same GCP project, with no differentiation between development, staging (or pre-production) and production environments. Ideally, there should be different projects and artifacts for each of these deployment stages, so that the production environment is the one that integrates with other production applications, and is only affected after all the changes have been tested thoroughly on the development and staging stages. Priority: must have.

### Model registry

No model versioning or model/experiment tracking was used on this solution. A Model Registry should be used to keep track of different model versions and to leverage remote storage. Using a Model Registry would require several architecture changes on the solution, since preprocessing and training jobs should download/update artifacts on this registry and deployment jobs should download the correct artifact versions for deployment. Priority: must have.

### Preprocessing and training jobs

Both of these tasks are now done together on the same script, which is either run locally or on a GA runner. These jobs should be split and set up to run remotely, downloading/uploading artifacts from/to GCS buckets or model registries. Having these jobs modularized also enables triggering them on a schedule or with more specialized triggers generated via Model Monitoring. Priority: must have.

### Data, Model and API Monitoring

Several solutions should be implemented to monitor the data and model in order to detect drifts and evaluate the performance of the model on real time. Detecting drifts can help trigger rollbacks or retraining jobs.

Monitoring the API is needed to trigger alerts in case of errors or abnormal system loads or usage. Priority: must have.

### Improved exploration and choice of model

Many features were left out of the analysis on the exploration notebook, and several decisions were made without justification or thorough analysis. Proper data analysis, feature engineering and selection and model experimentation needs to be done to guarantee if better results can be achieved and a better model can be deployed. Priority: nice to have.

### Other refactors

Throughout the document we described various decisions that were not optimal, like the API initialization or the model tracking. Implementing all of these will lead to a more robust and organized solution. Priority: nice to have.
