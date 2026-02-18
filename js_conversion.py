import os
import shutil
import sys
import subprocess
import keras

# --- CONFIGURATION ---
KERAS_MODEL_FILE = "best_poke_model.keras"
SAVED_MODEL_DIR = "temp_tf_saved_model"  # Directory for the intermediate graph
OUTPUT_DIR = "tfjs_model"

def step_1_export_graph():
    print(f"Step 1: Loading {KERAS_MODEL_FILE} (Keras 3)...")
    model = keras.models.load_model(KERAS_MODEL_FILE)
    
    # CRITICAL CHANGE: We use .export() instead of .save()
    # This creates a pure TensorFlow "SavedModel" (Graph format).
    # It strips away the Keras-specific metadata (like 'InputLayer' and 'nodeData')
    # that causes conflicts between Keras 3 and TFJS.
    print(f"Step 1b: Exporting to TF SavedModel (Graph) format...")
    model.export(SAVED_MODEL_DIR)
    print(f"  -> Exported to directory: {SAVED_MODEL_DIR}")

def step_2_convert_to_tfjs():
    print("Step 2: Running tensorflowjs_converter...")
    
    # We convert from "tf_saved_model" instead of "keras"
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format", "tf_saved_model",
        SAVED_MODEL_DIR,
        OUTPUT_DIR
    ]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("  -> Conversion Failed!")
        print(result.stderr)
        raise RuntimeError("tensorflowjs_converter failed.")
    
    print(f"  -> Conversion successful. Files saved to /{OUTPUT_DIR}")

# Step 3 (Patching) is REMOVED. 
# Graph models are compiled math operations, not JSON config objects.
# They do not suffer from the "batch_shape" or "nodeData" bugs.

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # cleanup previous output
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
            
        step_1_export_graph()
        step_2_convert_to_tfjs()
        
        print("\nSUCCESS! You can now use 'tfjs_model' in the browser.")
        
        # Clean up temp directory
        if os.path.exists(SAVED_MODEL_DIR):
            print(f"Cleaning up temp directory {SAVED_MODEL_DIR}...")
            shutil.rmtree(SAVED_MODEL_DIR)
            
    except Exception as e:
        print(f"\nERROR: {e}")