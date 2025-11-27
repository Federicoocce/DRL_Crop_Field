import os
import pickle
import sys

# Define the directory where the datasets are stored
HOME_DIR = os.path.expanduser('~')
DATASET_DIR = os.path.join(HOME_DIR, 'ros2_ws', 'drl_datasets', 'imitation_learning')

# Define the names of the input and output files
EXPERT_DATA_1_NAME = '180_auxiliary_sincurved.pkl'
EXPERT_DATA_2_NAME = '180_auxiliary.pkl'
VAL_DATA_NAME = '180_auxiliary_straight.pkl' # We'll check for this file but won't modify it
TRAIN_DATA_OUTPUT_NAME = '180_auxiliary_sincurved_curved.pkl'

def combine_and_prepare_datasets():
    """
    Loads two expert datasets, combines them into a single training dataset,
    and saves the result. It assumes a separate validation set already exists.
    """
    print("="*60)
    print("--- Starting Dataset Combination Script ---")
    print(f"Searching for datasets in: {DATASET_DIR}")
    print("="*60)

    # Construct full paths
    expert_path_1 = os.path.join(DATASET_DIR, EXPERT_DATA_1_NAME)
    expert_path_2 = os.path.join(DATASET_DIR, EXPERT_DATA_2_NAME)
    val_path = os.path.join(DATASET_DIR, VAL_DATA_NAME)
    train_output_path = os.path.join(DATASET_DIR, TRAIN_DATA_OUTPUT_NAME)

    # --- Load Datasets with Error Handling ---
    datasets_to_load = {
        'expert_1': expert_path_1,
        'expert_2': expert_path_2,
        'validation': val_path
    }
    loaded_data = {}

    for name, path in datasets_to_load.items():
        try:
            print(f"Loading '{os.path.basename(path)}'...")
            with open(path, 'rb') as f:
                loaded_data[name] = pickle.load(f)
            print(f"  -> Success. Found {len(loaded_data[name])} data points.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Dataset file not found at '{path}'")
            print("Please ensure all required dataset files exist before running.")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: An error occurred while loading '{path}': {e}")
            sys.exit(1)

    # --- Combine the Training Datasets ---
    print("\nCombining training datasets...")
    expert_dataset_1 = loaded_data['expert_1']
    expert_dataset_2 = loaded_data['expert_2']

    combined_train_dataset = expert_dataset_1 + expert_dataset_2

    print(f"Size of '{EXPERT_DATA_1_NAME}': {len(expert_dataset_1)}")
    print(f"Size of '{EXPERT_DATA_2_NAME}': {len(expert_dataset_2)}")
    print(f"  -> Total size of new training dataset: {len(combined_train_dataset)}")

    # --- Save the New Combined Training Dataset ---
    print(f"\nSaving combined training data to '{TRAIN_DATA_OUTPUT_NAME}'...")
    try:
        with open(train_output_path, 'wb') as f:
            pickle.dump(combined_train_dataset, f)
        print(f"  -> Successfully saved.")
    except Exception as e:
        print(f"FATAL ERROR: Could not save the new training dataset: {e}")
        sys.exit(1)

    print("\n--- Summary ---")
    print(f"Training Set ('{TRAIN_DATA_OUTPUT_NAME}'): {len(combined_train_dataset)} points (Combined)")
    print(f"Validation Set ('{VAL_DATA_NAME}'): {len(loaded_data['validation'])} points (Unchanged)")
    print("="*60)
    print("Process complete. You can now run the training launch file.")
    print("="*60)


if __name__ == '__main__':
    combine_and_prepare_datasets()