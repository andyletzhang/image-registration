from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from sklearn.neighbors import NearestNeighbors


def transform_points(
    points: np.ndarray, scale: float, rotation_angle: float, tx: float, ty: float
) -> np.ndarray:
    """
    Apply transformation (scale, rotation, translation) to a point set.

    Args:
        points: Array of shape (n, 2) containing the point coordinates
        scale: Scaling factor
        rotation_angle: Rotation angle in degrees
        tx: Translation in x direction
        ty: Translation in y direction

    Returns:
        Transformed points array of shape (n, 2)
    """
    # Apply scale
    scaled_points = points * scale

    # Apply rotation
    theta = np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    rotated_points = np.dot(scaled_points, rotation_matrix.T)

    # Apply translation
    translated_points = rotated_points + np.array([tx, ty])

    return translated_points


def _alignment_error(
    params: List[float], source_points: np.ndarray, target_points: np.ndarray
) -> float:
    """
    Calculate the alignment error for a given transformation.

    Args:
        params: Transformation parameters [scale, rotation_angle, tx, ty]
        source_points: Source point set of shape (n, 2)
        target_points: Target point set of shape (m, 2)

    Returns:
        Mean squared error between transformed source points and their nearest neighbors in target
    """
    scale, rotation_angle, tx, ty = params

    # Transform source points using current parameters
    transformed_source = transform_points(source_points, scale, rotation_angle, tx, ty)

    # Find nearest neighbors in target set
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(target_points)
    distances, _ = nbrs.kneighbors(transformed_source)

    # Calculate mean squared error
    mse = np.mean(distances**2)
    return mse


def find_optimal_transformation(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_guess: Optional[List[float]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Find the optimal transformation parameters to align source points with target points.

    Args:
        source_points: Array of shape (n, 2) containing the source point coordinates
        target_points: Array of shape (m, 2) containing the target point coordinates
        initial_guess: Optional initial guess for [scale, rotation_angle, tx, ty]
        verbose: Whether to print progress information

    Returns:
        Optimal parameters [scale, rotation_angle, tx, ty] as numpy array

    Raises:
        ValueError: If either point set contains fewer than 3 points
    """
    # Validate input
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError(
            "Both source and target point sets must contain at least 3 points"
        )

    # Initial guess if not provided
    if initial_guess is None:
        # Estimate initial scale based on the range of coordinates
        source_range = np.max(source_points, axis=0) - np.min(source_points, axis=0)
        target_range = np.max(target_points, axis=0) - np.min(target_points, axis=0)

        # Handle potential division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            range_ratio = target_range / source_range
            range_ratio = np.where(np.isfinite(range_ratio), range_ratio, 1.0)
            initial_scale = np.mean(range_ratio)

        # Start with no rotation and translation that aligns centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        initial_tx = target_centroid[0] - initial_scale * source_centroid[0]
        initial_ty = target_centroid[1] - initial_scale * source_centroid[1]

        initial_guess = [initial_scale, 0.0, initial_tx, initial_ty]

    # Set bounds for parameters
    bounds = [
        (0.1, 10.0),  # scale: reasonable range
        (-180.0, 180.0),  # rotation: full range in degrees
        (-5000, 5000),  # tx: wide range
        (-5000, 5000),  # ty: wide range
    ]

    # Optimize to find the best parameters
    result = optimize.minimize(
        _alignment_error,
        initial_guess,
        args=(source_points, target_points),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not result.success and verbose:
        print("Warning: Optimization did not converge.")

    return result.x


def iterative_registration_with_pruning(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 10,
    distance_threshold_percentile: float = 80,
    min_inlier_fraction: float = 0.3,
    inlier_shrink_factor: float = 0.9,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iteratively refines the point set registration by pruning outliers.

    Args:
        source_points: Original source point set of shape (n, 2)
        target_points: Original target point set of shape (m, 2)
        max_iterations: Maximum number of refinement iterations
        distance_threshold_percentile: Percentile for distance-based outlier rejection
        min_inlier_fraction: Minimum fraction of points to keep as inliers
        inlier_shrink_factor: Factor to multiply the threshold by in each iteration
        verbose: Whether to print progress information

    Returns:
        Tuple containing:
        - final transformation parameters [scale, rotation_angle, tx, ty]
        - source_inliers: indices of inlier points in the source set
        - target_inliers: subset of target points corresponding to inliers

    Raises:
        ValueError: If either point set contains fewer than 3 points
    """
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError(
            "Both source and target point sets must contain at least 3 points"
        )

    # Start with all points
    current_source = source_points.copy()
    current_target = target_points.copy()

    # Keep track of original indices
    source_indices = np.arange(len(source_points))

    # Initial transformation
    params = find_optimal_transformation(
        current_source, current_target, verbose=verbose
    )

    for iteration in range(max_iterations):
        # Apply current transformation
        transformed_source = transform_points(current_source, *params)

        # Find corresponding points in target set
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(current_target)
        distances, target_indices = nbrs.kneighbors(transformed_source)
        distances = distances.flatten()

        # Determine distance threshold for this iteration
        current_threshold = np.percentile(distances, distance_threshold_percentile) * (
            inlier_shrink_factor**iteration
        )

        # Identify inliers
        inlier_mask = distances < current_threshold

        # Ensure we keep at least min_inlier_fraction of the points
        if np.sum(inlier_mask) < min_inlier_fraction * len(current_source):
            # If pruning would remove too many points, keep the best min_inlier_fraction points
            num_to_keep = max(
                int(min_inlier_fraction * len(current_source)), 3
            )  # Keep at least 3 points
            inlier_indices = np.argsort(distances)[:num_to_keep]
            inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
            inlier_mask[inlier_indices] = True

        # If we're not removing any points, we can stop
        if np.all(inlier_mask):
            break

        # Update source and target points for next iteration
        current_source = current_source[inlier_mask]
        source_indices = source_indices[inlier_mask]
        corresponding_target_indices = target_indices[inlier_mask].flatten()
        current_target = current_target[corresponding_target_indices]

        # Recompute transformation with inliers only
        if (
            len(current_source) >= 3
        ):  # Need at least 3 points for a meaningful transformation
            new_params = find_optimal_transformation(
                current_source, current_target, initial_guess=params, verbose=verbose
            )
            params = new_params
        else:
            if verbose:
                print(
                    "Warning: Too few inliers remaining. Using last good transformation."
                )
            break

        if verbose:
            print(
                f"Iteration {iteration + 1}: {len(current_source)} inliers remaining "
                f"({len(current_source) / len(source_points) * 100:.1f}% of original)"
            )

    # Return final parameters and inlier indices
    return params, source_indices, current_target


def calculate_inverse_transform(params: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a transformation.

    Args:
        params: Transformation parameters [scale, rotation_angle, tx, ty]

    Returns:
        Inverse transformation parameters
    """
    scale, rotation, tx, ty = params

    # Inverse scale and rotation
    inv_scale = 1.0 / scale
    inv_rotation = -rotation

    # Calculate inverse translation (more complex due to rotation and scaling)
    theta = np.radians(rotation)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    inv_translation = np.array([-tx, -ty])
    inv_translation = inv_translation @ rotation_matrix / scale

    return np.array([inv_scale, inv_rotation, inv_translation[0], inv_translation[1]])


def register_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Performs bidirectional iterative registration to find the best alignment.
    This approach runs the iterative registration in both directions and selects the better result.

    Args:
        source_points: Numpy array of source points of shape (n, 2)
        target_points: Numpy array of target points of shape (m, 2)
        max_iterations: Maximum number of refinement iterations
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
        - 'scale': scaling factor
        - 'rotation_angle': rotation angle in degrees
        - 'tx': translation in x direction
        - 'ty': translation in y direction
        - 'inliers': indices of inlier points in the source or target set
        - 'is_source_to_target': boolean indicating if source->target direction was chosen
        - 'score': final alignment score (lower is better)
        - 'error': mean distance between corresponding points
        - 'inlier_ratio': fraction of points kept as inliers
    """
    # Try source to target
    source_to_target_params, source_inliers, _ = iterative_registration_with_pruning(
        source_points, target_points, max_iterations, verbose=verbose
    )

    # Calculate final error
    transformed_source = transform_points(
        source_points[source_inliers], *source_to_target_params
    )
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_points)
    source_to_target_distances, _ = nbrs.kneighbors(transformed_source)
    source_to_target_error = np.mean(source_to_target_distances)
    source_to_target_inlier_ratio = len(source_inliers) / len(source_points)

    # Try target to source (inverse transformation)
    target_to_source_params, target_inliers, _ = iterative_registration_with_pruning(
        target_points, source_points, max_iterations, verbose=verbose
    )

    # Calculate inverse error
    transformed_target = transform_points(
        target_points[target_inliers], *target_to_source_params
    )
    nbrs = NearestNeighbors(n_neighbors=1).fit(source_points)
    target_to_source_distances, _ = nbrs.kneighbors(transformed_target)
    target_to_source_error = np.mean(target_to_source_distances)
    target_to_source_inlier_ratio = len(target_inliers) / len(target_points)

    # Compare results with a composite score (lower is better)
    # We consider both alignment error and inlier ratio
    source_to_target_score = source_to_target_error / source_to_target_inlier_ratio
    target_to_source_score = target_to_source_error / target_to_source_inlier_ratio

    if verbose:
        print(
            f"Source->Target: Error={source_to_target_error:.2f}, "
            f"Inliers={source_to_target_inlier_ratio:.2f}, Score={source_to_target_score:.2f}"
        )
        print(
            f"Target->Source: Error={target_to_source_error:.2f}, "
            f"Inliers={target_to_source_inlier_ratio:.2f}, Score={target_to_source_score:.2f}"
        )

    # Choose the better direction
    if source_to_target_score <= target_to_source_score:
        if verbose:
            print("Selected Source->Target transformation as the better fit")
        scale, rotation, tx, ty = source_to_target_params
        result = {
            "scale": scale,
            "rotation_angle": rotation,
            "translation": np.array([ty, tx]),
            "inliers": source_inliers,
            "is_source_to_target": True,
            "score": source_to_target_score,
            "error": source_to_target_error,
            "inlier_ratio": source_to_target_inlier_ratio,
        }
    else:
        if verbose:
            print("Selected Target->Source transformation as the better fit")
        # Convert target->source parameters to source->target
        scale, rotation, tx, ty = calculate_inverse_transform(target_to_source_params)
        result = {
            "scale": scale,
            "rotation_angle": rotation,
            "translation": np.array([ty, tx]),
            "inliers": target_inliers,
            "is_source_to_target": False,
            "score": target_to_source_score,
            "error": target_to_source_error,
            "inlier_ratio": target_to_source_inlier_ratio,
        }

    return result


def visualize_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    params: Dict[str, Any],
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize the alignment of point sets before and after transformation.

    Args:
        source_points: Source point set of shape (n, 2)
        target_points: Target point set of shape (m, 2)
        params: Transformation parameters [scale, rotation_angle, tx, ty]
        save_path: Optional path to save the figure

    Returns:
        Transformed source points
    """
    scale = params["scale"]
    rotation_angle = params["rotation_angle"]
    ty, tx = params["translation"]
    inliers = params["inliers"]
    transformed_source = transform_points(source_points, scale, rotation_angle, tx, ty)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Before alignment
    ax1.scatter(
        target_points[:, 0],
        target_points[:, 1],
        c="blue",
        alpha=0.5,
        label="Target Points",
    )
    ax1.scatter(
        source_points[:, 0],
        source_points[:, 1],
        c="red",
        alpha=0.5,
        label="Source Points",
    )
    ax1.set_title("Before Alignment")
    ax1.legend()
    ax1.grid(True)

    # After alignment
    ax2.scatter(
        target_points[:, 0],
        target_points[:, 1],
        c="blue",
        alpha=0.5,
        label="Target Points",
    )
    ax2.scatter(
        transformed_source[:, 0],
        transformed_source[:, 1],
        c="red",
        alpha=0.5,
        label="Transformed Source",
    )

    # Highlight inliers
    ax2.scatter(
        transformed_source[inliers, 0],
        transformed_source[inliers, 1],
        facecolor="none",
        alpha=1,
        s=90,
        marker="o",
        edgecolors="green",
        linewidths=1,
        label="Inlier Points",
    )

    ax2.set_title("After Alignment")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# Manual Keypoint Registration
def compute_similarity_transform(src_pts, tgt_pts):
    """Computes scale, rotation, and translation to align src_pts to tgt_pts."""
    src_pts, tgt_pts = np.array(src_pts), np.array(tgt_pts)
    
    # Compute centroids
    centroid_src = np.mean(src_pts, axis=0)
    centroid_tgt = np.mean(tgt_pts, axis=0)

    # Center keypoints
    src_centered = src_pts - centroid_src
    tgt_centered = tgt_pts - centroid_tgt

    # Compute optimal rotation using SVD
    U, _, Vt = np.linalg.svd(src_centered.T @ tgt_centered)
    R = Vt.T @ U.T

    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute optimal scale
    scale = np.sum(tgt_centered * (src_centered @ R.T)) / np.sum(src_centered**2)

    # Compute translation
    translation = centroid_tgt - scale * (R @ centroid_src)

    return scale, R, translation
    
# Cellpose Registration
def get_cellpose_masks(img, channels=[0, 0]):
    from importlib.metadata import version
    from cellpose import models
    from cellpose.utils import remove_edge_masks

    cellpose_version = int(version("cellpose").split(".")[0])

    if cellpose_version < 4:
        cp_model = models.CellposeModel(gpu=True, model_type="cyto3")
        size_model = models.SizeModel(
            cp_model, pretrained_size=models.size_model_path("cyto3")
        )

        diam, _ = size_model.eval(img, channels=channels)
        masks, flows, styles = cp_model.eval(img, diameter=diam, channels=channels)
    
    else:
        cp_model = models.CellposeModel(gpu=True)

        masks, flows, styles = cp_model.eval(img)
    masks = remove_edge_masks(masks)

    return masks


def centroids(masks):
    from scipy.ndimage import center_of_mass

    return np.array(
        center_of_mass(masks, labels=masks, index=np.arange(masks.max()) + 1)
    )


def masks_to_alignment(source_masks, target_masks, show=False):
    source_centroids = centroids(source_masks)
    target_centroids = centroids(target_masks)
    out = register_points(source_centroids, target_centroids)

    if show:
        visualize_alignment(source_centroids, target_centroids, out)

    return out


def align_with_cellpose(source_img, target_img, verbose=False, show=False):
    if verbose:
        print("Getting source masks...", end="")
    source_masks = get_cellpose_masks(source_img)
    if verbose:
        print(f"done, found {source_masks.max()} cells in source image.")
        print("Getting target masks...", end="")
    target_masks = get_cellpose_masks(target_img)

    if verbose:
        print(f"done, found {target_masks.max()} cells in target image.")
        print("Registering centroids...")
    out = masks_to_alignment(source_masks, target_masks, show=show)

    return out
