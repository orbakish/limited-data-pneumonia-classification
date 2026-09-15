"""PyTorch Dataset and DataLoader over the committed split file lists.

Reads a list produced by src/splits.py, loads each image, applies the transform it is given,
and yields (image, label) pairs. It does not decide which images to use and it does not decide
which transform to apply - those belong to src/splits.py and src/augment.py respectively, so
this module stays the same for all 27 experiments.

Import it:

    from src.data import ChestXrayDataset, build_dataloader

or run it as a smoke check against the committed splits:

    python -m src.data
"""

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.splits import load_split

# PNEUMONIA is the positive class (1). This is not arbitrary: precision, recall and F1 are all
# defined relative to the positive class, so flipping this silently changes what those numbers
# mean in the report.
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}

class ChestXrayDataset(Dataset):
    """Dataset over an explicit list of image paths, labelled by their parent folder.

    paths is a list of Path as returned by src.splits.load_split; transform is applied to each
    loaded image and may be None, in which case the grayscale PIL image itself is returned.
    """

    def __init__(self, paths, transform=None):
        # Kept public so later code (Grad-CAM examples) can map a dataset index back to its file.
        self.paths = list(paths)
        self.transform = transform
        # Labels are resolved once, here, rather than on every __getitem__: a file sitting in an
        # unexpected folder then fails with a KeyError at construction, not halfway through an epoch.
        self.labels = [LABELS[path.parent.name] for path in self.paths]

    def __len__(self):
        """Return the number of images, which is what tells the DataLoader how long an epoch is."""
        return len(self.paths)

    def __getitem__(self, index):
        """Load one image and return (image, label).

        The label comes from the folder the file sits in - the layout is <split>/<class>/<file>,
        so the class is never stored separately and cannot fall out of sync with the file list.
        """
        path = self.paths[index]
        # Image.open is lazy: it reads the header and keeps the file open. convert() forces the
        # pixels to load and returns a new, independent image, so leaving the `with` block closes
        # the file - without that, worker processes can run out of file handles.
        with Image.open(path) as source:
            # "L" is PIL's single-channel 8-bit grayscale mode. Converting here, before any
            # transform, removes colour mode as a shortcut: every RGB image in this dataset is
            # PNEUMONIA, so a model that could see mode could learn the label without the lungs.
            image = source.convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def seed_worker(worker_id):
    """Seed Python's and NumPy's random generators inside one DataLoader worker process.

    PyTorch already gives every worker its own torch seed, derived from the DataLoader's
    generator, but it leaves the random and numpy modules alone. Any transform drawing from
    those would otherwise not be reproducible across runs. worker_id is unused but required:
    the DataLoader always passes it.
    """
    # Inside a worker, torch.initial_seed() returns the seed PyTorch assigned to that worker.
    # % 2**32 folds it into the range numpy's seed accepts.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(split_name, batch_size, transform=None, shuffle=False, seed=None, num_workers=0):
    """Build a DataLoader over one named split.

    split_name is a stem such as "val" or "train_20". Returns a torch DataLoader. Worker seeding
    is set here rather than left to the caller, because augmented runs are only reproducible if
    every worker process starts from a known seed.
    """
    dataset = ChestXrayDataset(load_split(split_name), transform)

    generator = None
    if seed is not None:
        # A dedicated generator: the DataLoader draws its shuffle order and each worker's base
        # seed from this, instead of from torch's global random state, which any other code in
        # the program could advance between runs.
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        # pin_memory copies each batch into page-locked RAM, which speeds up the later
        # .to("cuda") transfer. It does nothing useful without a GPU, so it is only on when one exists.
        pin_memory=torch.cuda.is_available(),
    )


def main():
    """Smoke-check the dataset and the loader against the committed splits."""
    dataset = ChestXrayDataset(load_split("train_20"))
    print(f"train_20: {len(dataset)} images")
    # The list is sorted, so index 0 is a NORMAL image and the last index a PNEUMONIA one.
    for index in (0, len(dataset) - 1):
        image, label = dataset[index]
        print(f"  [{index}] {dataset.paths[index].name}  size={image.size}  mode={image.mode}  label={label}")

    # A throwaway transform, imported here because nothing else in this module needs it. Its only
    # job is making images of different sizes stackable into one batch; the real preprocessing
    # belongs to src/augment.py and is not what this checks.
    from torchvision import transforms
    check_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # next(iter(loader)) pulls exactly one batch without looping over the whole epoch.
    images, labels = next(iter(build_dataloader("train_20", 8, check_transform, shuffle=True, seed=42)))
    print(f"batch: images {tuple(images.shape)}  labels {labels.tolist()}")

    # The reproducibility claim, tested directly: a second loader with the same seed must yield
    # the same first batch. torch.equal is True only if shapes and every value match.
    again, _ = next(iter(build_dataloader("train_20", 8, check_transform, shuffle=True, seed=42)))
    print(f"same seed gives same batch: {torch.equal(images, again)}")


if __name__ == "__main__":
    main()

