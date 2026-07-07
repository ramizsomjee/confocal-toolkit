"""Headless proof that preview affine and export rotation agree."""
import numpy as np
from scipy.ndimage import rotate as ndrotate, affine_transform
from confocal_toolkit import geometry as G


def test_matrix_matches_scipy():
    rng = np.random.default_rng(0)
    img = rng.random((51, 61))
    ctr = G.in_centre(img.shape)
    for ang in (0, 15, 30, 45, 90, 137, -22):
        R = G.rotation_matrix(ang)
        offset = ctr - R @ ctr
        mine = affine_transform(img, R, offset=offset, order=1,
                                mode="constant", cval=0, prefilter=False)
        ref = ndrotate(img, ang, reshape=False, order=1,
                       mode="constant", cval=0, prefilter=False)
        err = np.abs(mine - ref).max()
        assert err < 1e-6, f"angle {ang}: max err {err}"
    print("[ok] rotation_matrix reproduces scipy.ndimage.rotate (reshape=False)")


def test_world_to_output_invariant():
    """A marker's napari world coord must land on its actual oriented pixel."""
    H, W = 240, 300
    for ang in (0, 20, 45, 90, -35, 150):
        # bright marker at a few data locations
        for (r, c) in [(60, 90), (30, 250), (200, 40), (120, 150)]:
            img = np.zeros((1, H, W), np.float32)
            img[0, r, c] = 1.0

            # where napari shows this pixel (world coords)
            A = G.preview_affine((H, W), ang)
            world = A @ np.array([r, c, 1.0])
            world = world[:2]

            oriented, offset = G.orient_image(img, ang)
            # actual location of the marker in the oriented array
            yx = np.unravel_index(np.argmax(oriented[0]), oriented[0].shape)
            predicted = world + offset

            d = np.hypot(*(np.array(yx) - predicted))
            assert d <= 1.5, (f"angle {ang} pt ({r},{c}): predicted {predicted} "
                              f"actual {yx} dist {d:.2f}")
    print("[ok] world->output invariant holds (box on screen crops right tissue)")


def test_box_crop_lands_on_marker():
    """End-to-end: place a box centred on a marker in world coords, crop from
    the oriented image, and confirm the marker is inside the crop centre."""
    H, W = 260, 340
    r, c = 80, 120
    for ang in (0, 25, 60, -40):
        img = np.zeros((1, H, W), np.float32)
        img[0, r - 1:r + 2, c - 1:c + 2] = 1.0   # 3x3 block survives interp
        A = G.preview_affine((H, W), ang)
        world = (A @ np.array([r, c, 1.0]))[:2]
        oriented, offset = G.orient_image(img, ang)
        y0, y1, x0, x1 = G.world_box_to_output(world, 40, offset,
                                               oriented.shape[1:])
        crop = oriented[0, y0:y1, x0:x1]
        assert crop.max() > 0.5, f"angle {ang}: marker not in crop"
        yx = np.unravel_index(np.argmax(crop), crop.shape)
        # marker should be near the crop centre (within a few px)
        assert abs(yx[0] - 20) <= 3 and abs(yx[1] - 20) <= 3, (ang, yx)
    print("[ok] box centred in world crops the marker at the crop centre")


if __name__ == "__main__":
    test_matrix_matches_scipy()
    test_world_to_output_invariant()
    test_box_crop_lands_on_marker()
    print("\nGEOMETRY OK")
