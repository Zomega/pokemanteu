import os
import sys

# Add current directory to sys.path so the dummy module is found
sys.path.append(os.getcwd())
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflowjs as tfjs

# Load the trained .keras file
model = keras.models.load_model("best_poke_model.keras")

# Export to TF.js format (requires 'pip install tensorflowjs')
# This creates a folder 'tfjs_model' you can upload to your web server
tfjs.converters.save_keras_model(model, "tfjs_model")