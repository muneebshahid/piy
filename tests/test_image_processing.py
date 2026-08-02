"""Tests for image processing before model submission."""

import base64
from io import BytesIO

import pytest
from PIL import Image

from tile.tools.support.image_processing import (
    ImageProcessingError,
    ImageProcessingOptions,
    ProcessedImage,
    process_image,
)
from tile.types.tools import ImageMimeType


def _jpeg(width: int, height: int) -> bytes:
    """Return JPEG bytes for a solid-color image."""

    image = Image.new("RGB", (width, height), "red")
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    """Return JPEG bytes containing an EXIF orientation tag."""

    image = Image.new("RGB", (width, height), "red")
    exif = Image.Exif()
    exif[274] = orientation
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)
    return output.getvalue()


def _png(width: int, height: int) -> bytes:
    """Return PNG bytes for a deterministic image."""

    image = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), ((x * 17) % 256, (y * 31) % 256, (x * y) % 256))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _processed(result: ProcessedImage | ImageProcessingError) -> ProcessedImage:
    """Return a processed image from a pipeline result."""

    assert isinstance(result, ProcessedImage)
    return result


def test_process_image_applies_jpeg_exif_orientation() -> None:
    """Rotate JPEG bytes according to their EXIF orientation tag."""

    original = _jpeg_with_orientation(width=2, height=3, orientation=6)

    processed = _processed(process_image(original, "image/jpeg"))

    with Image.open(BytesIO(processed.data)) as image:
        assert processed.mime_type == "image/jpeg"
        assert image.size == (3, 2)
        assert image.getexif().get(274) is None
        assert processed.was_resized is False


@pytest.mark.parametrize(
    ("original", "mime_type"),
    [
        pytest.param(_jpeg(width=2, height=3), "image/jpeg", id="jpeg-no-exif"),
        pytest.param(_png(width=2, height=3), "image/png", id="small-png"),
    ],
)
def test_process_image_preserves_fitting_image_bytes(
    original: bytes,
    mime_type: ImageMimeType,
) -> None:
    """Leave image bytes unchanged when they fit all limits with no EXIF work."""

    processed = _processed(process_image(original, mime_type))

    assert processed.data == original
    assert processed.mime_type == mime_type


def test_process_image_resizes_to_dimension_limits() -> None:
    """Resize images to fit within max width and height."""

    original = _png(width=8, height=4)

    processed = _processed(
        process_image(
            original,
            "image/png",
            ImageProcessingOptions(max_width=4, max_height=4),
        )
    )

    assert processed.width == 4
    assert processed.height == 2
    assert processed.was_resized is True
    assert processed.dimension_note() == (
        "[Image: original 8x4, displayed at 4x2. "
        "Multiply coordinates by 2.00 to map to original image.]"
    )


def test_process_image_reduces_until_base64_payload_fits() -> None:
    """Progressively reduce dimensions until the base64 payload fits."""

    original = _png(width=64, height=64)

    processed = _processed(
        process_image(
            original,
            "image/png",
            ImageProcessingOptions(
                max_width=64,
                max_height=64,
                max_base64_bytes=800,
            ),
        )
    )

    assert processed.width < 64
    assert processed.height < 64
    assert len(base64.b64encode(processed.data)) < 800


def test_process_image_raises_when_no_candidate_fits() -> None:
    """Raise a processing error when even the smallest image is too large."""

    original = _png(width=8, height=8)

    with pytest.raises(
        ImageProcessingError,
        match="could not be resized below the inline image size limit",
    ):
        process_image(
            original,
            "image/png",
            ImageProcessingOptions(max_width=1, max_height=1, max_base64_bytes=1),
        )


def test_process_image_reports_decode_failure_with_cause() -> None:
    """Chain the underlying decode error instead of claiming a size failure."""

    with pytest.raises(ImageProcessingError, match="could not be processed") as error:
        process_image(b"not an image", "image/png")

    assert isinstance(error.value.__cause__, OSError)


def test_process_image_wraps_decompression_bomb_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap Pillow decompression-bomb rejections as processing errors."""

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(ImageProcessingError, match="could not be processed") as error:
        process_image(_png(width=8, height=8), "image/png")

    assert isinstance(error.value.__cause__, Image.DecompressionBombError)
