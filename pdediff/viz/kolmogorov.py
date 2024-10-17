
r"""Kolmogorov experiment helpers"""

import os
from pathlib import Path
from typing import *

from PIL import Image, ImageDraw, ImageOps
import seaborn

from numpy.typing import ArrayLike
import numpy as np


def vorticity2rgb(
    w: ArrayLike,
    vmin: float = -1.25,
    vmax: float = 1.25,
) -> ArrayLike:
    w = np.asarray(w)
    w = (w - vmin) / (vmax - vmin)
    w = 2 * w - 1
    w = np.sign(w) * np.abs(w) ** 0.8
    w = (w + 1) / 2
    w = seaborn.cm.icefire(w)
    w = 256 * w[..., :3]
    w = w.astype(np.uint8)

    return w


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 4,
    zoom: int = 1,
    **kwargs,
) -> Image.Image:
    w = vorticity2rgb(w, **kwargs)
    m, n, width, height, _ = w.shape

    img = Image.new(
        'RGB',
        size=(
            n * (width + pad) + pad,
            m * (height + pad) + pad
        ),
        color=(255, 255, 255),
    )

    for i in range(m):
        for j in range(n):
            offset = (
                j * (width + pad) + pad,
                i * (height + pad) + pad,
            )

            img.paste(Image.fromarray(w[i][j]), offset)

            if mask is not None:
                img.paste(
                    Image.new('L', size=(width, height), color=240),
                    offset,
                    Image.fromarray(~mask[i][j]),
                )

    if zoom > 1:
        return img.resize((img.width * zoom, img.height * zoom), resample=0)
    else:
        return img


def sandwich(
    w: ArrayLike,
    offset: int = 5,
    border: int = 1,
    mirror: bool = False,
    **kwargs,
):
    w = vorticity2rgb(w, **kwargs)
    n, width, height, _ = w.shape

    if mirror:
        w = w[:, :, ::-1]

    img = Image.new(
        'RGB',
        size=(
            width + (n - 1) * offset,
            height + (n - 1) * offset,
        ),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(img)

    for i in range(n):
        draw.rectangle(
            (i * offset - border, i * offset - border, img.width, img.height),
            (255, 255, 255),
        )
        img.paste(Image.fromarray(w[i]), (i * offset, i * offset))

    if mirror:
        return ImageOps.mirror(img)
    else:
        return img


def save_gif(
    w: ArrayLike,
    file: Path,
    dt: float = 0.2,
    **kwargs,
) -> None:
    w = vorticity2rgb(w, **kwargs)

    imgs = [Image.fromarray(img) for img in w]
    imgs[0].save(
        file,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * dt),
        loop=0,
    )
