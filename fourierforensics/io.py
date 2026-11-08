"""Bounded, source-preserving image input."""

import warnings
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass
class LoadedImage:
    pixels: np.ndarray
    source_bytes: bytes
    warnings: list[str]
    format: str


def load_image(data: bytes) -> LoadedImage:
    """Keep original size, grid, orientation and source bytes for analysis."""
    if len(data) > 25 * 1024 * 1024:
        raise ValueError("Upload exceeds 25 MiB")
    notes = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as im:
                if im.width * im.height > 16_000_000 or min(im.size) < 32:
                    raise ValueError("Use images with sides >=32 pixels and <=16 megapixels")
                if im.getexif().get(274, 1) != 1:
                    notes.append("EXIF rotation was not applied; original pixel grid is analyzed.")
                if im.info.get("icc_profile"):
                    notes.append("ICC profile is not applied; decoded RGB values are analyzed.")
                if getattr(im, "n_frames", 1) > 1:
                    notes.append("Only the first frame is analyzed.")
                fmt = im.format or "unknown"
                if "A" in im.getbands() or "transparency" in im.info:
                    rgba = im.convert("RGBA")
                    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), rgba)
                    notes.append("Transparency was composited on white.")
                elif im.mode not in ("RGB", "L"):
                    notes.append(
                        f"{im.mode} input converted to 8-bit RGB; conversion may affect artifacts."
                    )
                pixels = np.array(im.convert("RGB"))
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Cannot decode a supported image safely") from exc
    return LoadedImage(pixels, data, notes, fmt)
