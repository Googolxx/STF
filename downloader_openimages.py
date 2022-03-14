import fiftyone

if __name__ == '__main__':
    """Download the training/test data set from OpenImages."""

    dataset_train = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="train",
        max_samples=300000,
        label_types=["classifications"],
        dataset_dir='openimages',
    )
    dataset_test = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="test",
        max_samples=10000,
        label_types=["classifications"],
        dataset_dir='openimages',
    )
