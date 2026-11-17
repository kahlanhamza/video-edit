"""Regression: median_flow_shift survives cv2's (N, 1, 2) point arrays.

The Eustis real-footage sweep crashed at the second sampled frame with
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
because `np.median(disp, axis=0)` on cv2's (N, 1, 2) flow output yields (1, 2)
whose elements are arrays. numpy 2.x refuses float() on those. The helper
flattens to (N, 2) first — this test pins both the shape contract and the
numeric contract (median, not mean).
"""

import numpy as np
from director_os.vision_fx.compute.local_onnx import median_flow_shift


def test_cv2_point_shape_survives():
    # exactly what cv2.calcOpticalFlowPyrLK hands back after boolean indexing
    disp = np.array([[3.0, 1.0], [5.0, 3.0], [1.0, -1.0], [3.0, 1.0]]).reshape(4, 1, 2)
    dx, dy = median_flow_shift(disp)
    assert (dx, dy) == (3.0, 1.0)


def test_flat_shape_and_odd_count_use_true_median():
    disp = np.array([[0.0, 0.0], [10.0, 0.0], [4.0, 2.0]])
    dx, dy = median_flow_shift(disp)
    assert (dx, dy) == (4.0, 0.0)


def test_single_displacement():
    dx, dy = median_flow_shift(np.array([[[-7.5, 2.5]]]))
    assert (dx, dy) == (-7.5, 2.5)
