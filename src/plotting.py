import matplotlib.pyplot as plt
import numpy as np
from .io_backends import TifHandler, ParquetHandler




#TODO: plotting as for now assumes squre images , adjust for non square
def plot_band(handler,band_index:int, ax=None, figsize=(6,6)):
    if isinstance(handler, TifHandler):
        band_data = handler.get_band(band_index)
    elif isinstance(handler, ParquetHandler):
        band_col = f"band_{band_index}"
        band_data = handler.data[band_col].to_numpy().reshape(
            int(np.sqrt(len(handler.data[band_col]))),
            int(np.sqrt(len(handler.data[band_col])))
        )
    else:
        raise ValueError("Unsupported handler type")
    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure(figsize=figsize)
    plt.imshow(band_data, cmap='gray')
    plt.colorbar()
    plt.title(f'Band {band_index}')
    plt.show()


def plot_composite(
    handler,
    bands: tuple[int, int, int],
    stretch: bool = True,
    scale: float = 1.0,
    ax=None
):
    imgs = []
    for b in bands:
        if isinstance(handler, TifHandler):
            data = handler.get_band(b).astype(float)
        elif isinstance(handler, ParquetHandler):
            df = handler.get_bands([b])
            width = df["x"].max() + 1
            height = df["y"].max() + 1
            data = df[f"band_{b}"].to_numpy().reshape(height, width).astype(float)
        else:
            raise TypeError("Handler must be TifHandler or ParquetHandler")

        data = data / scale

        if stretch:
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
        imgs.append(data)

    rgb = np.stack(imgs, axis=-1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(f"Composite bands {bands} (scale={scale})")
    ax.axis('off')
    return ax






