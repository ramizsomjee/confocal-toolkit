"""Rotation geometry shared by the live viewer and the export step.

The one invariant that makes the tool correct: the affine we hand napari for
the live preview and the array we produce with scipy for export must describe
the *same* rotation, so a box drawn on screen (in world coordinates) crops the
exact tissue it covers.

Conventions
-----------
* All coordinates are (row, col) == (y, x) to match numpy / napari.
* ``R(angle)`` is the rotation matrix such that
  ``scipy.ndimage.affine_transform(img, R, offset)`` reproduces
  ``scipy.ndimage.rotate(img, angle, reshape=False)`` — verified in tests.
* Preview affine pins the input centre to itself (no expand shift) so boxes
  placed in world coordinates stay on the same tissue as the angle changes.
* Export uses ``scipy.ndimage.rotate(..., reshape=True)`` (expands the canvas,
  centres the content, no data loss). A world coordinate maps to an output
  pixel by ``world - in_centre + out_centre``.
"""

from __future__ import annotations

import numpy as np


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """2x2 rotation matrix in (row, col) matching scipy.ndimage.rotate.

    scipy.ndimage.affine_transform maps output->input; using this matrix as the
    ``matrix`` argument reproduces ``rotate(img, angle_deg, reshape=False)``.
    """
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    # (row, col) form equivalent to scipy's rotate for the (0, 1) axes plane.
    return np.array([[c, s], [-s, c]])


def in_centre(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.array([(h - 1) / 2.0, (w - 1) / 2.0])


def preview_affine(shape: tuple[int, int], angle_deg: float) -> np.ndarray:
    """3x3 data->world affine for napari (rotation about the image centre).

    The image centre maps to itself for every angle, so world coordinates are
    stable and boxes do not drift when the rotation slider moves.
    """
    # napari maps data->world; that is the inverse of scipy's output->input R.
    fwd = rotation_matrix(angle_deg).T           # forward (data->world) rotation
    ctr = in_centre(shape)
    translate = ctr - fwd @ ctr
    affine = np.eye(3)
    affine[:2, :2] = fwd
    affine[:2, 2] = translate
    return affine


def orient_image(stack: np.ndarray, angle_deg: float):
    """Rotate a (C, H, W) stack to the oriented frame (expand, no data loss).

    Returns ``(oriented, world_to_output_offset)`` where oriented is
    (C, H', W') and ``output_pixel = world_coord + world_to_output_offset``.
    """
    from scipy.ndimage import rotate as ndrotate

    assert stack.ndim == 3, f"expected (C,H,W), got {stack.shape}"
    in_ctr = in_centre(stack.shape[1:])
    chans = [ndrotate(stack[c], angle_deg, reshape=True, order=1,
                      mode="constant", cval=0, prefilter=False)
             for c in range(stack.shape[0])]
    oriented = np.stack(chans, axis=0)
    out_ctr = in_centre(oriented.shape[1:])
    offset = out_ctr - in_ctr
    return oriented, offset


def world_box_to_output(center_world, size_px: float, offset, out_shape):
    """Fixed-size square (centre in world coords) -> integer crop bounds.

    Returns ``(y0, y1, x0, x1)`` clamped inside ``out_shape`` (H', W'),
    keeping the box exactly ``size_px`` on a side where possible.
    """
    cy, cx = np.asarray(center_world, float) + np.asarray(offset, float)
    s = int(round(size_px))
    H, W = out_shape
    s = min(s, H, W)
    y0 = int(round(cy - s / 2.0))
    x0 = int(round(cx - s / 2.0))
    y0 = max(0, min(y0, H - s))
    x0 = max(0, min(x0, W - s))
    return y0, y0 + s, x0, x0 + s
