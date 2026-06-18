from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    transaction_date: float = Field(..., json_schema_extra={"example": 2013.25})
    house_age: float = Field(..., json_schema_extra={"example": 32.0})
    distance_to_mrt: float = Field(..., json_schema_extra={"example": 350.5})
    convenience_stores: int = Field(..., json_schema_extra={"example": 4})
    latitude: float = Field(..., json_schema_extra={"example": 24.98298})
    longitude: float = Field(..., json_schema_extra={"example": 121.54024})
