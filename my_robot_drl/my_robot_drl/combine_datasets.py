import os
import pickle
import sys
import gc
import joblib  # <--- NEW IMPORT

# Define the directory
HOME_DIR = os.path.expanduser('~')
DATASET_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_datasets', 'imitation_learning')

EXPERT_DATA_1_NAME = '180_auxiliary_sin_cs_mixed_ss.pkl'
EXPERT_DATA_2_NAME = '180_auxiliary_mixed_long.pkl'
TRAIN_DATA_OUTPUT_NAME = '180_auxiliary_sin_cs_mixed_ss_ml.pkl'

def get_size_mb(obj):
    """Recursively estimates size of objects (approximate)"""
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([get_size_mb(v) for v in obj.values()])
    elif isinstance(obj, list):
        size += sum([get_size_mb(x) for x in obj])
    return size

def combine_and_prepare_datasets():
    print("="*60)
    print("--- Starting Dataset Combination (Joblib Mode) ---")
    print(f"Dataset Dir: {DATASET_DIR}")
    print("="*60)

    expert_path_1 = os.path.join(DATASET_DIR, EXPERT_DATA_1_NAME)
    expert_path_2 = os.path.join(DATASET_DIR, EXPERT_DATA_2_NAME)
    train_output_path = os.path.join(DATASET_DIR, TRAIN_DATA_OUTPUT_NAME)

    # --- 1. Load First Dataset ---
    combined_data = []
    try:
        print(f"Loading '{EXPERT_DATA_1_NAME}'...")
        # We assume input files are still standard pickle. 
        # If they fail, change this to joblib.load(expert_path_1)
        with open(expert_path_1, 'rb') as f:
            combined_data = pickle.load(f)
        
        print(f"  -> Success. Loaded {len(combined_data)} points.")
    except Exception as e:
        print(f"FATAL ERROR loading first file: {e}")
        sys.exit(1)

    # --- 2. Load Second Dataset and Append ---
    try:
        print(f"Loading '{EXPERT_DATA_2_NAME}'...")
        with open(expert_path_2, 'rb') as f:
            temp_data = pickle.load(f)
            print(f"  -> Success. Loaded {len(temp_data)} points.")
            
            print("Merging datasets...")
            combined_data.extend(temp_data)
            
            print("Freeing temp memory...")
            del temp_data
            gc.collect()

    except Exception as e:
        print(f"FATAL ERROR loading second file: {e}")
        sys.exit(1)

    total_len = len(combined_data)
    print(f"  -> Total size: {total_len} points")

    # --- 3. Save with Joblib ---
    print(f"\nSaving to '{TRAIN_DATA_OUTPUT_NAME}' using Joblib...")
    
    try:
        # Joblib handles large numpy arrays much better than pickle.
        # compress=3 offers a good balance of speed and size.
        joblib.dump(combined_data, train_output_path, compress=3)
        
        print(f"  -> Successfully saved.")
        del combined_data
        gc.collect()
        
    except Exception as e:
        print(f"FATAL ERROR: Could not save dataset: {e}")
        sys.exit(1)

    print("\n--- Process Complete ---")

if __name__ == '__main__':
    combine_and_prepare_datasets()