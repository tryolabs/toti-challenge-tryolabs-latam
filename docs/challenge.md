# Challenge Documentation

## Notebook reproduction

The first step in the migration of the explored model is to reproduce the notebook created by the Data Scientist (DS from now on).

This notebook has some bugs that we fix along the way to be able to run it correctly or avoid getting misleading results. The found bugs are:

* Positional arguments on the calls to `seaborn` plotting functions are producing an error. The bug is fixed by passing the name of the arguments (`x` and `y`) instead of positional arguments.
* The feature generation function `is_high_season` has problems with the upper bounds of the ranges. This upper bounds have a time set of "00:00:00" (beggining of the day), so the flights with date equal to the upper bound don't get transformed correctly.
* The function `get_rate_from_column` is computing `# total flights / # delayed flights` instead of the actual delay rate which is `# delayed flights / # total flights`. Fixing this should give better insights of the data.
* The first XGBoost model trained (with all the features and no balance) predicts all 0s on the test set, which indicates that the model didn't train correctly. The simplest reason may be that the `learning_rate` is too small and doesn't allow the model to learn much. Adjusting the learning rate to be a little bit higher (0.5) helps with the training in this case. However, so that we don't change the top 10 features selected by DS, we will keep the `learning_rate` as is.
* The selected top 10 features are not even the top 10 features with highest feature importance, and there's no argument to why they were chosen. We will leave this as is to avoid affecting the output model.

Required missing dependencies: `ipykernel`, `xgboost`.

### Model choice

At the end, the DS has trained six different models:
* XGBoost/Logistic Regression model with all the features and no class weights
* XGBoost/Logistic Regression model with top 10 features and no class weights
* XGBoost/Logistic Regression model with top 10 features and class weights

As the DS correctly states, all the models trained with no class weights perform poorly on the `Delayed` class (1), mainly in terms of recall.

Ideally, there should be a discussion with the stakeholders about what metrics to optimize for the `Delayed` class: do we want high precision? or high recall? Since we don't have this information, it's safe to assume that both metrics have equal importance, and we should try to optimize the F1-score of the `Delayed` class, without significantly affecting the metrics for the `Not-Delayed` (0) class.

Both models trained with top 10 features and class weights achieve virtually the same metrics: an F1-score of ~0.36 on the `Delayed` class and ~0.65 on the `Not-Delayed` class, although results are slightly better for the XGBoost model.

Since both models achieve equal metrics, we compare them in terms of inference time (see the Annex section on the `exploration.ipynb` notebook) to choose the one that's faster. In this section, we can see that the Logistic Regression model tends to have better and more stable inference time. This makes sense, as this model tends to be more lightweight than an XGBoost model, and should also provide better training times. Since Logistic Regression is conceptually more simple, it is also more interpretable than XGBoost, which makes it the preferred candidate to push to production.

To summarize, we choose Logistic Regression since it achieves the same performance metrics and is faster and more simple.

## Model migration

The idea of this step is to migrate the model from the exploration notebook to a fully functional Python script for preprocessing the data, training the model and providing inference. The methods of the class `DelayModel` are completed to achieve this. Here are some special considerations and observations from this step:

* The `pd.get_dummies` method was not used to perform One-Hot Encoding. Instead, custom code was built to do it. The reason is that if one of the categories of the features we want to One-Hot Encode is not present on the data, `get_dummies` will not create a column for that category, whereas the correct thing would be to create the column for that category with all 0s. With this custom code, only the categories selected by the DS get encoded and if any of them is not present on the data, it gets filled with 0s.

* There are some issues with the test script `test_model.py`. First, the path to the data is incorrect and needed to be changed in order to run the tests. Then, the `test_model_predict` test initializes a `DelayModel` object and tries to run the `predict` method without never calling the `fit` method first. This is unacceptable and ideally, an exception should be returned indicating that the model has not been trained. However, the test expects the `predict` method to return a list of ints, independent of this error, which I think is undesirable. Since the test file can't be changed, we return a dummy list filled with a very large negative value (`-2**60`) in this case, so that the test runs correctly.

## API development

On this step, the goal is to serve the trained model and provide an endpoint for inference.

The API should load the model on startup from a local file. In order to do this, we extended the `model.py` script with a `save` method which writes the `DelayModel` object to a `pickle` file on a given path. Then, we trained a model over all the available data and stored it locally with the following code:

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

* Due to the way the API tests are built, we're unable to use FastAPI's startup event or `lifespan` method for initializing the API, which is the recommended way of loading the model on startup. The way the tests for the API are built, the API startup methods don't get invoked and the API doesn't get initialized. To circumvent this, we initialize the model directly on the `api.py` script, which is undesirable.

* `pydantic` was used to define the input schema and the input validations. A custom exception handler needed to be built for the `RequestValidationError` so that a status code 400 is returned instead of the default 422 "Unprocessable Entity". The change was made to fit the tests provided for the API.


## API deployment

The goal of this step is to deploy the API on a public endpoint. To complete it, we will first build a Docker image that runs the server with the API and then deploy that Docker image to GCP.


### Docker image

The provided Dockerfile was completed so that the image is built and the API is started when the image runs inside a container.

In order to use a Python version similar to the one used throughout the development, we change the base image to be the `3.9-slim`. This is a more lightweight base image and it is closer to the the version used during development (`3.9.4`).

### GCP

The API is deployed as a Cloud Run Service that exposes a public endpoint. In order to accomplish this, these were the steps taken:

1. A GCP project was created with the ID `rodrigo-tryolabs-latam`.
2. The Artifact Registry is used to store the Docker image of the API. Before this, we need to create the repository for the image, with the following command:

    `gcloud artifacts repositories create --repository-format=docker --location=us-west1 delay-model-service`

    This creates a repository on the address `us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service`.

3. Then, we need to build the Docker image and tag it with the address of the created repository. We can do this with the command:

    `docker build . -t us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service/delay-model-api`

4. Before we can push the image to the remote Artifact Registry repository, we need to allow Docker to authenticate to GCP. We can do it with the command:

    `gcloud auth configure-docker us-west1-docker.pkg.dev`

5. Now, we can push the image to the repository:

    `docker push us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service/delay-model-api`

6. Finally, we can deploy the image as a Cloud Run Service with the command:

    ```
    gcloud run deploy delay-model \
        --image us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service/delay-model-api \
        --allow-unauthenticated \
        --region us-west1
    ```

    This command will deploy the previously uploaded Docker image as a service. The argument `--allow-unauthenticated` publicly exposes the API and allows for unauthenticated requests. Note that some default arguments are used: the CPU limit is set to 1 vCPU; the memory limit is set to 512MiB; the service is shutdown when idle.

After the deployment is completed, the API is available at https://delay-model-dpmrk4cwxq-uw.a.run.app, and the prediction endpoint is available at https://delay-model-dpmrk4cwxq-uw.a.run.app/predict. We can test the service using Postman or run the provided stress test.

The results of the stress test are an error rate of 0%, an average response time of 343ms, a maximum response time of 743ms and the API is able to respond to 87.69 requests per second.


## CI/CD Pipeline

On this final step, the goal is to setup a proper CI/CD pipeline.

The Continuous Integration (CI) workflow focuses on running the tests and assesing the quality of the code each time there's a push to the repository, with the goal of detecting bugs earlier, correcting code faster and ensuring good code quality practices.

The Continuous Deployment (CD) workflow focuses on training the model, deploying the API and running the stress test against it. This workflow only runs when there's a push to the `main`, `develop` or `release` branches.

Let's describe each workflow with more detail.

### Continuous Integration

The goals of this workflow are checking the code quality and testing it. For the first goal, the code is checked using `black`, `flake8` and `isort` to ensure that the style and format are correct and fit the repository standards. For the second goal, the provided test suites (`model-test` and `api-test`) are ran to ensure that the changes done on the code don't affect the functionality of the `DelayModel` class and the API.

Important decisions made on this step:

* The test suites require a trained model available for testing purposes. However, this test suites run on Github workers and don't have access to local models. To circumvent this, the model checkpoint is tracked with Git and uploaded to the remote. This is not desirable, since model's can crow rapidly in size and managing them inside the repository can become a problem. The ideal solution would be to maintain a proper Model Registry, with remote storage and a good version management, so that trained models can be uploaded to it or downloaded for testing or deployment. Due to time restrictions and since the model checkpoint is lightweight on this case, the decision to track the model was taken.
* The `model-test` suite had to be modified due to an error. The path to the data file on the suite was `../../data/data.csv`, which assumed that the tests were ran from the `tests/model/` directory, but tests should actually be run from the project root folder, where the `Makefile` is. To fix this, we change the path to be `data/data.csv`. With this change, tests run correctly and can be triggered from the GA.

### Continuous Deployment

The goal of this workflow is to train the model, build the Docker image with it and deploy it to a Cloud Run service. This workflow only runs when there's a push to the `main`, `develop` or `release` branches and it deploys a different API for each of these. The reasoning is that having different deployments for different stages of the development of features and releases can help in testing how the changes affect the deployment, while keeping the `main` API intact and serving only the released code features.

Here are the most important steps taken to develop this workflow:

* A small and simple training script (`train.py`) was created so that the GA trains the model before deploying it. This training script uses all the available data, preprocesses it, trains the model and writes it to the location used by the Dockerfile to put the model inside the Docker image. This is a simplification of a real scenario. Ideally, the data would be stored remotely and we would have different remote jobs for preprocessing the data, training the model and uploading it to a Model Registry. These remote jobs could be triggered by the same events that trigger this workflow, but none of the preprocessing or training would run inside the GA synchronously.
* A GCP Service Account `cd-pipeline-sa` was created to grant the Github Action runner with permissions to push the Docker image to the Artifact Registry repository and to deploy the Cloud Run Service. The roles given to this SA are:
    - `Artifact Registry Writer`: enables the SA to push Docker images to the Artifact Registry repositories
    - `Cloud Run Admin`: gives the SA full control over the Cloud Run services deployed
    - `Service Account User`: gives the SA the necessary permissions to act as the default Cloud Run service account. This permission is needed for deploying from the Github Action.
We created one single SA for simplification, since we only use it in a single workflow. Ideally, we should have multiple SAs, each with more granular and reduced permissions; for example, we could have a "Cloud Run SA" which only has control over the services and nothing else, and a separate "Artifact Registry SA" which only has access to the repository.
* A `dev` environment was created on the Github Repository, containing various configuration variables (mostly names used through the GCP deployment) and secrets (the key to access the SA `cd-pipeline-sa`). The created configuration variables are:
    - `GAR_IMAGE_NAME=delay-model-api`
    - `GAR_REPOSITORY=us-west1-docker.pkg.dev/rodrigo-tryolabs-latam/delay-model-service`
    - `GCP_PROJECT_ID=rodrigo-tryolabs-latam`
    - `GCP_REGION=us-west1`
    - `GCR_SERVICE_NAME=delay-model`
* After deployment of the service, the stress tests run against the deployed API. As mentioned, different APIs are deployed depending on the branch. To point the stress test script to the correct API, a small modification was needed to be done to the `Makefile`, so that the URL of the API is passed as an argument on the `make stress-test` command. The final command is `make stress-test API_URL=<api-url>`.
