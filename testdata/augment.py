from torchvision import transforms

preproc = transforms.Compose(
    [
        lambda image: image.convert("RGB"),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
)


def transform(sample):
    sample["ppm"] = preproc(sample["png"])
    del sample["png"]
    return sample
