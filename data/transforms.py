from monai.transforms import (
    Compose,
    LoadImaged,
    )
train_transforms = Compose([
    LoadImaged(keys='image')
])
test_transforms = Compose([
    LoadImaged(keys='image')
])