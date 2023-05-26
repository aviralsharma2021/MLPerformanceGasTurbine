import pandas as pd
import numpy as np
import tensorflow as tf
from .apps import ApiConfig
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from django.http import JsonResponse
import json


# # Create your views here.
class PerfPrediction(APIView):
    def post(self, request):
        data = request.data
        alt = data['alt']
        mach = data['mach']
        zxn = data['zxn']

        model = ApiConfig.model
        inputs = np.array([[alt, mach, zxn]]).astype(np.float64)

        cols = ["Thrust (kN)", "TSFC (g/kN.s)", "EGT (K)", "T2 (K)", "T3 (K)", "P2 (kPa)",
            "P3 (kPa)", "Wf (kg/s)", "St8Tt (K)"]
        
        outdict = pd.DataFrame(model.predict(inputs), columns=cols).to_dict(orient='records')
        return Response(outdict)
