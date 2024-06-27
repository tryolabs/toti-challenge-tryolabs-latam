from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from challenge.model import DelayModel

model = None


app = FastAPI()


# The recommended way to load the model is to do it on a startup event.
# However, due to the way the API tests are built, we can't use startup events.
def load_model() -> DelayModel:
    # Load the model from a temporary location. This path should be read from a config file.
    model = DelayModel.load("challenge/tmp/model_checkpoint.pkl")

    assert isinstance(model, DelayModel)
    return model


model = load_model()


# This custom exception handler returns a 400 status code instead of the default 422
@app.exception_handler(RequestValidationError)
async def custom_request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


class Flight(BaseModel):
    # Representation of an input flight
    operator: str = Field(alias="OPERA", description="Flight operator")
    flight_type: Literal["N", "I"] = Field(
        alias="TIPOVUELO",
        description="Flight type. N for national, I for international.",
    )
    month: int = Field(
        alias="MES", ge=1, le=12, description="Month of the operation of flight"
    )


class Input(BaseModel):
    # Representation of a list of flights for batch processing
    flights: list[Flight] = Field(min_items=1)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(input: Input) -> dict:
    # Convert the input into a DataFrame
    data = pd.DataFrame(data=input.dict(by_alias=True)["flights"])

    # Preprocess the data
    data = model.preprocess(data)

    # Provide inference
    pred = model.predict(data)

    return {"predict": pred}
