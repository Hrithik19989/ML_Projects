from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    transaction_date: float = Field(
        ..., alias="X1 transaction date", example=2013.250
    )
    house_age: float = Field(..., alias="X2 house age", example=32.0)
    distance_to_mrt: float = Field(
        ..., alias="X3 distance to the nearest MRT station", example=84.878
    )
    convenience_stores: int = Field(
        ..., alias="X4 number of convenience stores", example=10
    )
    latitude: float = Field(..., alias="X5 latitude", example=24.98298)
    longitude: float = Field(..., alias="X6 longitude", example=121.54024)

    class Config:
        populate_by_name = True
