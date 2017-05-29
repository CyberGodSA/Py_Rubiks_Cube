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
    d_point = Rpts - view
    d_point_view = np.dot(d_point, view).reshape(d_point.shape[:-1] + (1,))
    d_proj = -d_point * v2 / d_point_view

    trans = [i for i in range(1, d_proj.ndim)] + [0]
    return np.array([np.dot(d_proj, x_dir),
                     np.dot(d_proj, y_dir),
                     -np.dot(d_point, z_dir)]).transpose(trans)
