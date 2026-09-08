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

    # Gerçek resim klasörü (iç içe klasör)
    image_root = next(root.rglob("MILK10k_Training_Input/MILK10k_Training_Input"))

    pairs = []

    # Aynı lesion_id'ye ait clinical ve dermoscopic görüntüleri eşleştir
    for lesion_id, group in dataset.groupby("lesion_id"):

        clinical = group[group["image_type"].str.contains("clinical", case=False)]
        derm = group[group["image_type"].str.contains("dermoscopic", case=False)]

        if clinical.empty or derm.empty:
            continue

        clinical_row = clinical.iloc[0]
        derm_row = derm.iloc[0]

        clinical_path = image_root / lesion_id / f"{clinical_row['isic_id']}.jpg"
        derm_path = image_root / lesion_id / f"{derm_row['isic_id']}.jpg"

        if not (clinical_path.exists() and derm_path.exists()):
            continue

        # One-hot kolonlardan etiketi bul
        label = next(
            col for col in CLASS_COLUMNS
            if clinical_row[col] == 1
        )

        pairs.append({
            "lesion_id": lesion_id,
            "clinical_path": str(clinical_path),
            "dermoscopic_path": str(derm_path),
            "label": label,
            "age_approx": clinical_row["age_approx"],
            "sex": clinical_row["sex"],
            "skin_tone_class": clinical_row["skin_tone_class"],
            "site": clinical_row["site"],
        })

    return pairs