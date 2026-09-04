
from pathlib import Path
import pandas as pd

CLASS_COLUMNS = [
    "AKIEC","BCC","BEN_OTH","BKL","DF",
    "INF","MAL_OTH","MEL","NV","SCCKA","VASC"
]

def load_dataset(data_path):

    data_path = Path(data_path)

    metadata = pd.read_csv(data_path/"MILK10k_Training_Metadata.csv")

    groundtruth = pd.read_csv(data_path/"MILK10k_Training_GroundTruth.csv")

    dataset = metadata.merge(
        groundtruth,
        on="lesion_id",
        how="left"
    )

    return dataset


def create_pairs(dataset,data_path):

    data_path = Path(data_path)

    image_files = list(
        (data_path/"MILK10k_Training_Input").rglob("*.jpg")
    )

    image_map = {
        img.stem:str(img)
        for img in image_files
    }

    clinical = dataset[
        dataset["image_type"]=="clinical: close-up"
    ].copy()

    dermoscopic = dataset[
        dataset["image_type"]=="dermoscopic"
    ].copy()

    clinical["clinical_path"] = clinical["isic_id"].map(image_map)

    dermoscopic["dermoscopic_path"] = dermoscopic["isic_id"].map(image_map)

    paired = clinical.merge(
        dermoscopic[["lesion_id","dermoscopic_path"]],
        on="lesion_id"
    )

    paired["label"] = paired[CLASS_COLUMNS].idxmax(axis=1)
    

    return paired