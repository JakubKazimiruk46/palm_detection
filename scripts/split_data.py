import json
import logging
import os
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_dataset(
    input_file: str, output_dir: str, ratios: tuple = (0.7, 0.2, 0.1)
) -> None:
    """
    Splits the dataset into train, test, and subsample sets.
    """
    if sum(ratios) != 1.0:
        logger.error("Split ratios must sum to 1.0")
        return

    try:
        with open(input_file) as f:
            content = f.read().strip()
        if not content.startswith("{"):
            content = "{" + content + "}"
        data = json.loads(content)
        logger.info(f"Successfully loaded {len(data)} entries.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return

    users_map: dict[str, list[str]] = {}
    for img_id, annotation in data.items():
        user_id = annotation.get("user_id", "unknown")
        if user_id not in users_map:
            users_map[user_id] = []
        users_map[user_id].append(img_id)

    all_users = list(users_map.keys())
    random.shuffle(all_users)

    total_users = len(all_users)
    train_end = int(total_users * ratios[0])
    test_end = train_end + int(total_users * ratios[1])

    train_users = all_users[:train_end]
    test_users = all_users[train_end:test_end]
    subsample_users = all_users[test_end:]

    def flatten_ids(user_list):
        ids = []
        for uid in user_list:
            ids.extend(users_map[uid])
        return ids

    splits = {
        "train": flatten_ids(train_users),
        "test": flatten_ids(test_users),
        "subsample": flatten_ids(subsample_users),
    }

    for key in splits:
        random.shuffle(splits[key])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "split.json")

    try:
        with open(output_path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Split configuration saved to {output_path}")
        logger.info(
            f"Users split -> Train: {len(train_users)}, Test: {len(test_users)}, Sub: {len(subsample_users)}"
        )
        logger.info(
            f"Images split -> Train: {len(splits['train'])}, Test: {len(splits['test'])}, Sub: {len(splits['subsample'])}"
        )
    except OSError as e:
        logger.error(f"Failed to save split file: {e}")


if __name__ == "__main__":
    split_dataset("annotations.json", "dataset")
