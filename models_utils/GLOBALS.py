import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Min/Max tensors
min_values_type1 = torch.tensor([-2.30336538209, -3.40176282264, -0.976451779318], device=device)
max_values_type1 = torch.tensor([7.2808711902, 3.9957190444, 4.76762666942], device=device)
min_values_type2 = torch.tensor([-19.603912, -19.594337, -19.603912], device=device)
max_values_type2 = torch.tensor([19.594337, 19.603912, 19.594337], device=device)

# Paths
BASE_DIR = r"C:\Users\husseien\Desktop\340915149_322754953\Source Code"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
files_directory = os.path.join(DATA_DIR, "unlabeled", "unlabeled")


# Mappings
activity_id_mapping = {
    'brushing_teeth': 0, 'idle': 1, 'preparing_sandwich': 2, 'reading_book': 3,
    'stairs_down': 4, 'stairs_up': 5, 'typing': 6, 'using_phone': 7,
    'using_remote_control': 8, 'walking_freely': 9, 'walking_holding_a_tray': 10,
    'walking_with_handbag': 11, 'walking_with_hands_in_pockets': 12,
    'walking_with_object_underarm': 13, 'washing_face_and_hands': 14,
    'washing_mug': 15, 'washing_plate': 16, 'writing': 17
}

id_activity_mapping = {v: k for k, v in activity_id_mapping.items()}
sensor_mapping = {'smartwatch': 0, 'vicon': 1}