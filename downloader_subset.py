import fiftyone

if __name__ == '__main__':
    dataset = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="test",
        max_samples=50000,
        label_types=["classifications"],
        dataset_dir='openimages',
    )

# if __name__ == '__main__':
#     dataset = fiftyone.zoo.load_zoo_dataset(
#         "open-images-v6",
#         split="train",
#         max_samples=2000,
#         label_types=["classifications"],
#         dataset_dir='gmmdata',
#     )