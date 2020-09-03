import numpy as np
import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, ValidationError, validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from api.model import Model, get_model, avail
from attentivefp.utils.chem import standardize_mol, smiles2mol, mol2smiles

import logging
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    smiles: List[str]

class PredictResponse(BaseModel):
    data: List[dict]
    status: str

class ModelDesc(BaseModel):
    key : str
    columns : List[str]
    name : Optional[str] = None

class ModelListResponse(BaseModel):
    models : List[ModelDesc]

app = FastAPI()

@app.post("/models/{model}", response_model=PredictResponse)
async def predict(input: PredictRequest, model: str, standardize : bool = False):
    model = get_model(model)
    smiles = np.array(input.smiles)
    logger.info(f'num smiles: {len(smiles)}')

    mols = [smiles2mol(s) for s in smiles]
    if standardize:
        mols = [standardize_mol(m) for m in mols]

    y_pred = model.predict(mols)
    # remove nan
    data = [{k: v for k, v in m.items() if pd.notnull(v)} for m in y_pred.to_dict(orient='rows')]
    result = PredictResponse(data=data, status="OK")
    return result

@app.get("/models", response_model=ModelListResponse, response_model_exclude_unset=True)
async def list_available():
    models = [ModelDesc(key=m, columns=c) for m, c in avail()]
    return ModelListResponse(models=models)