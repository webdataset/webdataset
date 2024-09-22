from torchvision import transforms

preproc = transforms.Compose(
    [
        lambda image: image.convert("RGB"),
        transforms.CenterCrop(224),
    ]
)


def transform(sample):
    sample["ppm"] = preproc(sample["png"])
    del sample["png"]
    return sample
