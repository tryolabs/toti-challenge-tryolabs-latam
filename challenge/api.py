from fastapi import FastAPI
from pickle import load
from challenge.model import DelayModel

model = None


app = FastAPI()


# The recommended way to load the model is to do it on a startup event.
# However, due to the way the API tests are built, we can't use startup events.
def load_model() -> DelayModel:
    # Load the model from a temporary location
    model = DelayModel.load("challenge/tmp/model_checkpoint.pkl")

    assert isinstance(model, DelayModel)
    return model


model = load_model()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    return
