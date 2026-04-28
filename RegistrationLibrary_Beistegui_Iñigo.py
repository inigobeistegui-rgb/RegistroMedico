import numpy as np
from scipy.spatial import KDTree


# Private helpers

def _centroid_prealign(target, source):
    """
    Translate source so its centroid coincides with target's.

    For high aspect-ratio objects (e.g. a sword, aspect ratio ~36) the
    initial centroid gap can exceed the entire width of the object.
    Every source point then matches the wrong region of the target mesh,
    and ICP converges to a local minimum with a large translation error.
    Pre-alignment collapses this gap to zero before the first iteration.
    """
    num_dimensions = source.shape[1]
    prealignment_transform = np.eye(num_dimensions + 1)
    prealignment_transform[:num_dimensions, num_dimensions] = (
        target.mean(axis=0) - source.mean(axis=0)
    )
    return prealignment_transform


# Public API

def calculate_distances_and_correspondences(
        target,
        source,
        max_correspondence_distance):
    """
    For every source point find its nearest neighbour in target.

    Only correspondences whose distance is below
    max_correspondence_distance are kept; the rest are discarded so
    that outliers do not corrupt the transform estimate.
    """
    search_tree = KDTree(target)
    distances, target_indices = search_tree.query(source)

    valid_mask = distances < max_correspondence_distance

    valid_source_indices = np.where(valid_mask)[0]
    valid_target_indices = target_indices[valid_mask]
    valid_distances = distances[valid_mask]

    correspondences = np.column_stack(
        (valid_source_indices, valid_target_indices)
    )

    return valid_distances, correspondences


def calculate_best_fit_transform(source, target, correspondences):
    """
    Compute the least-squares rigid transform that maps source onto
    target using the supplied point correspondences.

    Uses SVD on the cross-covariance matrix of the centred point sets.
    The determinant check prevents SVD from returning a reflection
    instead of a proper rotation when the point clouds are nearly
    co-planar.
    """
    source_points = source[correspondences[:, 0]]
    target_points = target[correspondences[:, 1]]

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    # Centring decouples rotation from translation: SVD finds the
    # optimal rotation on centred clouds, then translation is recovered
    # from the centroids.
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    cross_covariance = source_centered.T @ target_centered

    left_singular, _, right_singular_transposed = np.linalg.svd(
        cross_covariance
    )

    # If det(V * U^T) = -1 the SVD returned a reflection, not a
    # rotation. Flipping the sign of the last diagonal entry forces
    # det = +1 without affecting the other axes.
    reflection_correction = np.eye(left_singular.shape[0])
    reflection_correction[-1, -1] = np.linalg.det(
        right_singular_transposed.T @ left_singular.T
    )

    rotation = (
        right_singular_transposed.T
        @ reflection_correction
        @ left_singular.T
    )
    translation = target_centroid - rotation @ source_centroid

    num_dimensions = source.shape[1]
    transformation = np.eye(num_dimensions + 1)
    transformation[:num_dimensions, :num_dimensions] = rotation
    transformation[:num_dimensions, num_dimensions] = translation

    return transformation


def transform_points(points, transformation):
    """
    Apply a homogeneous transformation matrix to an array of points.

    Works for both 2-D (3x3 matrix) and 3-D (4x4 matrix) inputs.
    """
    num_dimensions = points.shape[1]
    homogeneous_points = np.hstack(
        [points, np.ones((points.shape[0], 1))]
    )
    transformed_points = homogeneous_points @ transformation.T
    return transformed_points[:, :num_dimensions]


def calculate_rmse(distances):
    """
    Root Mean Square Error of the supplied correspondence distances.
    """
    return float(np.sqrt(np.mean(distances ** 2)))


def icp(target,
        source,
        max_correspondence_distance=10,
        max_iterations=20,
        metric_delta_threshold=1e-6):
    """
    Iterative Closest Point algorithm.

    Aligns source to target by iteratively finding nearest-neighbour
    correspondences and computing the best-fit rigid transform.

    Centroid pre-alignment is applied once before the loop to remove
    the bulk translational gap, which is the main cause of convergence
    to a wrong local minimum on elongated objects.
    """
    working_source = source.copy()
    num_dimensions = source.shape[1]

    prealignment = _centroid_prealign(target, working_source)
    working_source = transform_points(working_source, prealignment)
    total_transformation = prealignment

    previous_rmse = float('inf')
    iteration_history = []

    for _ in range(max_iterations):

        distances, correspondences = calculate_distances_and_correspondences(
            target, working_source, max_correspondence_distance
        )

        # A minimum of D+1 correspondences is required to constrain a
        # rigid transform in D dimensions (e.g. 3 points in 2D).
        if len(correspondences) < num_dimensions + 1:
            break

        iteration_transform = calculate_best_fit_transform(
            working_source, target, correspondences
        )

        # Left-multiply to accumulate: the new transform is applied on
        # top of all previous ones, preserving the correct order.
        total_transformation = iteration_transform @ total_transformation
        working_source = transform_points(
            working_source, iteration_transform
        )

        current_rmse = calculate_rmse(distances)
        iteration_history.append(
            (current_rmse, total_transformation.copy())
        )

        # Stop when improvement drops below the convergence threshold.
        if abs(previous_rmse - current_rmse) < metric_delta_threshold:
            break

        previous_rmse = current_rmse

    return total_transformation, iteration_history