import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.put("/predict")
async def predict(alt, mach, zxn):
    df1 = pd.read_csv(file.file)
    file.file.close()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf
    import numpy as np

    inputs = np.array([[alt, mach, zxn]]).astype(np.float64)

    model = tf.keras.models.load_model("models/model")
    cols = ["Thrust (kN)", "TSFC (g/kN.s)", "EGT (K)", "T2 (K)", "T3 (K)", "P2 (kPa)",
            "P3 (kPa)", "Wf (kg/s)", "St8Tt (K)"]
    pd.DataFrame(model.predict(inputs), columns=cols).to_csv("output_infer.csv", index=False)

    return FileResponse("./output_infer.csv", filename="output_infer.csv")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3030)
