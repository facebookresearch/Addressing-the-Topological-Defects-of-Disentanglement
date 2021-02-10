"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from .xiaolinwu import draw_line

blue = (0, 0, 255)
yellow = (255, 255, 0)
white = (255, 255, 255)
black = (0, 0, 0)


def generate_images_from_coords(NPX, NP, C, cols):
    images = list()
    for c in range(C.shape[2]):
        img = Image.new("RGB", (NPX, NPX), white)
        for p in range(NP - 1):
            if (C[0, p + 1, c] != C[0, p, c]) or (C[1, p + 1, c] != C[1, p, c]):
                draw_line(
                    img,
                    (C[0, p + 1, c], C[1, p + 1, c]),
                    (C[0, p, c], C[1, p, c]),
                    cols[c],
                )
                draw_line(
                    img,
                    (C[0, p, c], C[1, p, c]),
                    (C[0, p + 1, c], C[1, p + 1, c]),
                    cols[c],
                )
        if (C[0, p + 1, c] != C[0, 0, c]) or (C[1, p + 1, c] != C[1, 0, c]):
            draw_line(
                img, (C[0, p + 1, c], C[1, p + 1, c]), (C[0, 0, c], C[1, 0, c]), cols[c]
            )
            draw_line(
                img, (C[0, 0, c], C[1, 0, c]), (C[0, p + 1, c], C[1, p + 1, c]), cols[c]
            )
        images.append(np.array(img))
    return images


# Draw images correspoding to different classes
def plot_and_save_grid(NPX, images, margin=1, name="FIGS/junk.png"):
    grid = np.zeros((NPX + 2 * margin, NPX * NC + margin * NC + margin, 3))
    pointer = 0
    for img in images:
        grid[
            margin : NPX + margin, 0 + pointer + margin : NPX + pointer + margin, :
        ] = img
        pointer += NPX + margin

    im = Image.fromarray(np.uint8((grid)))
    im.save(name)
    return im


class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, V, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.root = ts.root
        # self.transform = transforms.ToTensor()
        self.V = V

    def __len__(self):
        return len(self.V[0])

    def __getitem__(self, idx):
        try:
            return tuple([self.V[v][idx] for v in range(len(self.V))])
        except:
            pdb.set_trace()
        # return (self.transform(self.train_data[idx,:,:,:]),self.train_labels[idx])
        # return Dataset.__getitem__(self, idx)
        # super()


def pytorch_dataset(V, batch_size):
    # order = np.random.permutation(NS)
    ts = MyDataset(V)
    loader = torch.utils.data.DataLoader(ts, batch_size=batch_size, shuffle=True)
    return loader


def generate_dataset(NPX, NC, NP):

    NS = NC * 2  # number of samples

    # coordinates of each classes of objects
    C = np.random.randint(0 + NPX / 6, NPX - 1 - NPX / 6, (2, NP, NC))
    cols = np.zeros((NS, 3))

    # Generate images corresponding to different classes using Xiaolin Wu's line algorithm for anti-aliasing

    cols = np.zeros((NS, 3))
    X = np.array(
        generate_images_from_coords(NPX, NP, C[:, :, :].reshape((2, NP, NC)), cols)
    )

    X = 1 - np.mean(X, axis=3)
    # normalize (negative sign ensure background is min)
    X = X / -X.mean()

    y = np.arange(NC)
    y = y.flatten()
    Y = y.astype(int)

    split = NS // 4

    Xtrain = X[:split]
    Ytrain = Y[:split]
    Xtest = X[split:]
    Ytest = Y[split:]
    return ((Xtrain, Ytrain), (Xtest, Ytest))


def generate_angles(NT1, NT2, NC):

    # create pairs of shape with all angles
    NT = NT1 * NT2 ** 2
    [ind1, ind2] = np.meshgrid(range(NT), range(NT))
    s1 = ind1.flatten()
    s2 = ind2.flatten()
    alphas = (s1 - s2) % (NT1)

    sangle1 = np.floor(s1 / NT2 ** 2)
    sangle2 = np.floor(s2 / NT2 ** 2)

    strans1 = s1 % NT2 ** 2
    strans2 = s2 % NT2 ** 2

    stransx1 = np.floor(strans1 / NT2)
    stransx2 = np.floor(strans2 / NT2)

    stransy1 = strans1 % NT2
    stransy2 = strans2 % NT2

    alphas1 = (sangle1 - sangle2) % (NT1)
    alphas2 = (stransx1 - stransx2) % (NT2)
    alphas3 = (stransy1 - stransy2) % (NT2)

    s1_all_shapes = (
        np.tile(s1, (int(NC / 2)))
        + NT * np.tile(np.arange(int(NC / 2)).T, (NT * NT, 1)).T.flatten()
    )
    s2_all_shapes = (
        np.tile(s2, (int(NC / 2)))
        + NT * np.tile(np.arange(int(NC / 2)).T, (NT * NT, 1)).T.flatten()
    )

    alphas_all_shapes1 = np.tile(alphas1, int(NC / 2))
    alphas_all_shapes2 = np.tile(alphas2, int(NC / 2))
    alphas_all_shapes3 = np.tile(alphas3, int(NC / 2))

    alphas = (alphas1, alphas2, alphas3)
    alphas_all_shapes = (alphas_all_shapes1, alphas_all_shapes2, alphas_all_shapes3)

    return s1, s2, s1_all_shapes, s2_all_shapes, alphas, alphas_all_shapes


def x_to_image(x):
    """Takes a single input x and transforms it into image for im.show"""
    if x.dim() == 2:
        n_channels = 1
    else:
        n_channels = x.shape[0]
    n_pixels = x.shape[1]

    x_image = x.reshape(n_channels, n_pixels, n_pixels)
    x_image = x_image.permute(1, 2, 0)
    # sequeeze to remove in case of a singel channel
    x_image = x_image.squeeze()
    return x_image
