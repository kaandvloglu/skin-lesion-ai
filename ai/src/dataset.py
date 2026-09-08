from pathlib import Path
import pandas as pd

CLASS_COLUMNS = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]


def load_dataset(data_path):
    data_path = Path(data_path)

    metadata_path = Path("/kaggle/input/datasets/kaandevelioglu/milk10k-skin-lesion-dataset/MILK10K_Training_Metadata.csv")
    groundtruth_path = Path("/kaggle/input/datasets/kaandevelioglu/milk10k-skin-lesion-dataset/MILK10k_Training_GroundTruth.csv")

    metadata = pd.read_csv(metadata_path)
    groundtruth = pd.read_csv(groundtruth_path)

    dataset = metadata.merge(
        groundtruth,
        on="lesion_id",
        how="left"
    )

    return dataset


def create_pairs(dataset, data_path):
    data_path = Path(data_path)

    # Dataset içindeki bütün jpg dosyalarını bul
    image_files = list(data_path.rglob("*.jpg"))

    image_map = {
        img.stem: str(img)
        for img in image_files
    }

    clinical = dataset[
        dataset["image_type"] == "clinical: close-up"
    ].copy()

    dermoscopic = dataset[
        dataset["image_type"] == "dermoscopic"
    ].copy()

    clinical["clinical_path"] = clinical["isic_id"].map(image_map)
    dermoscopic["dermoscopic_path"] = dermoscopic["isic_id"].map(image_map)

    paired = clinical.merge(
        dermoscopic[["lesion_id", "dermoscopic_path"]],
        on="lesion_id"
    )

    paired["label"] = paired[CLASS_COLUMNS].idxmax(axis=1)

    # Eksik dosya yollarını temizle
    before = len(paired)

    paired = paired.dropna(
        subset=["clinical_path", "dermoscopic_path"]
    )

    after = len(paired)

    print(f"Total pairs: {before}")
    print(f"Valid pairs: {after}")
    print(f"Removed: {before-after}")

    return paired