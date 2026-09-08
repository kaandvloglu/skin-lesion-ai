from pathlib import Path
import pandas as pd

CLASS_COLUMNS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]


def load_dataset(data_path=None):
    root = Path("/kaggle/input")

    metadata_path = next(root.rglob("MILK10k_Training_Metadata.csv"))
    groundtruth_path = next(root.rglob("MILK10k_Training_GroundTruth.csv"))

    metadata = pd.read_csv(metadata_path)
    groundtruth = pd.read_csv(groundtruth_path)

    dataset = metadata.merge(
        groundtruth,
        on="lesion_id",
        how="left"
    )

    return dataset


def create_pairs(dataset, data_path=None):
    root = Path("/kaggle/input")

    image_root = next(root.rglob("MILK10k_Training_Input"))

    pairs = []

    for _, row in dataset.iterrows():
        lesion_folder = image_root / row["lesion_id"]

        if not lesion_folder.exists():
            continue

        for img in lesion_folder.glob("*.jpg"):
            pairs.append({
                "image_path": str(img),
                "metadata": row.to_dict()
            })

    return pairs