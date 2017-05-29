import numpy as np

from quaternion import *


def project_points(points, q, view, vertical=[0, 1, 0]):
    # project points using a quaternion q and a view v
    points = np.asarray(points)
    view = np.asarray(view)

    x_dir = np.cross(vertical, view).astype(float)

    if np.all(x_dir == 0):
        raise ValueError("vertical is parallel to v")

    x_dir /= np.sqrt(np.dot(x_dir, x_dir))

    # get the unit vector corresponing to vertical
    y_dir = np.cross(view, x_dir)
    y_dir /= np.sqrt(np.dot(y_dir, y_dir))

    # normalize the viewer location: this is the z-axis
    v2 = np.dot(view, view)
    z_dir = view / np.sqrt(v2)

    # rotate the points
    R = q.rotation_matrix()
    Rpts = np.dot(points, R.T)

    # project the points onto the view
    dpoint = Rpts - view
    dpoint_view = np.dot(dpoint, view).reshape(dpoint.shape[:-1] + (1,))
    dproj = -dpoint * v2 / dpoint_view

    trans = [i for i in range(1, dproj.ndim)] + [0]
    return np.array([np.dot(dproj, x_dir),
                     np.dot(dproj, y_dir),
                     -np.dot(dpoint, z_dir)]).transpose(trans)
