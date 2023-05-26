import os
import tensorflow as tf
from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    MODEL_FILE = os.path.join(settings.MODELS, 'model')
    model = tf.keras.models.load_model(MODEL_FILE)
