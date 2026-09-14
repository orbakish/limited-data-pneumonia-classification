"""PyTorch Dataset and DataLoader over the committed split file lists.

Reads a list produced by src/splits.py, loads each image, applies the transform it is given,
and yields (tensor, label) pairs. It does not decide which images to use and it does not decide
which transform to apply - those belong to src/splits.py and src/augment.py respectively, so
this module stays the same for all 27 experiments.

Not a script; import it:

    from src.data import ChestXrayDataset, build_dataloader
"""

from torch.utils.data import Dataset

# PNEUMONIA is the positive class (1). This is not arbitrary: precision, recall and F1 are all
# defined relative to the positive class, so flipping this silently changes what those numbers
# mean in the report.
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}

class ChestXrayDataset(Dataset):
    """Dataset over an explicit list of image paths, labelled by their parent folder.

    paths is a list of Path as returned by src.splits.load_split; transform is applied to each
    loaded image and may be None, in which case the raw PIL image is returned.
    """

    def __init__(self, paths, transform=None):
        raise NotImplementedError

    def __len__(self):
        """Return the number of images, which is what tells the DataLoader how long an epoch is."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Load one image and return (image, label).

        The label comes from the name of the folder the file sits in, via LABELS - the layout
        is <split>/<class>/<file>, so the class is never stored separately and cannot fall out
        of sync with the file list.
        """
        raise NotImplementedError

def build_dataloader(split_name, batch_size, transform=None, shuffle=False, seed=None, num_workers=0):
    """Build a DataLoader over one named split.

    split_name is a stem such as "val" or "train_20". Returns a torch DataLoader. Worker seeding
    is set here rather than by the caller, because augmented runs are only reproducible if every
    worker process starts from a known seed.
    """
    raise NotImplementedError

