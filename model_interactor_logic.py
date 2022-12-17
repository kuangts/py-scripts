import numpy as np

class ModelInteractorLogic:

    view_width_ = 1
    view_height_ = 1
    model_size_ = 20.0

    Camera::Transform matrix_
    geometry::AxisAlignedBoundingBox model_bounds_
    Eigen::Vector3f center_of_rotation_

    Camera::Transform matrix_at_mouse_down_
    Eigen::Vector3f center_of_rotation_at_mouse_down_
