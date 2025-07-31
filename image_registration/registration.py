import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import correlate2d
from scipy.spatial import cKDTree
from skimage.transform import SimilarityTransform, estimate_transform

# Cellpose Registration
def get_cellpose_masks(img, channels=[0, 0]):
    from importlib.metadata import version

    from cellpose import models
    from cellpose.utils import remove_edge_masks

    cellpose_version = int(version('cellpose').split('.')[0])

    if cellpose_version < 4:
        cp_model = models.CellposeModel(gpu=True, model_type='cyto3')
        size_model = models.SizeModel(cp_model, pretrained_size=models.size_model_path('cyto3'))

        diam, _ = size_model.eval(img, channels=channels)
        masks, flows, styles = cp_model.eval(img, diameter=diam, channels=channels)

    else:
        cp_model = models.CellposeModel(gpu=True)

        masks, flows, styles = cp_model.eval(img)
    masks = remove_edge_masks(masks)

    return masks


def centroids(masks):
    from scipy.ndimage import center_of_mass

    return np.array(center_of_mass(masks, labels=masks, index=np.arange(masks.max()) + 1))


def masks_to_alignment(moving_masks, fixed_masks, scale_factor, show=False, verbose=False, **kwargs):
    moving_centroids = centroids(moving_masks)
    fixed_centroids = centroids(fixed_masks)
    
    initial_translation = find_initial_translation(moving_centroids*scale_factor, fixed_centroids, bin_size=10) # TODO: automatically determine bin size
    print(f'Initial translation: {initial_translation}')
    initial_transform = SimilarityTransform(translation=initial_translation, scale=scale_factor)
    transform=compute_ICP(moving_centroids, fixed_centroids, initial_transform=initial_transform, verbose=verbose, **kwargs)

    if show:
        visualize_alignment(moving_centroids, fixed_centroids, transform)

    # Adjust the transform to (y, x) convention
    transform = SimilarityTransform(
        scale=transform.scale,
        rotation=-transform.rotation,
        translation=transform.translation[::-1]
    )
    
    return transform

def visualize_alignment(moving_points, fixed_points, transform):
    transformed_points = transform(moving_points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'aspect': 'equal'})
    axes[0].scatter(moving_points[:, 1], moving_points[:, 0], color='blue', label='Moving Points')
    axes[0].scatter(fixed_points[:, 1], fixed_points[:, 0], color='red', label='Fixed Points')
    axes[0].set_title('Original Points')
    axes[0].legend()
    axes[0].invert_yaxis()
    axes[1].scatter(transformed_points[:, 1], transformed_points[:, 0], color='blue', label='Transformed Moving Points')
    axes[1].scatter(fixed_points[:, 1], fixed_points[:, 0], color='red', label='Fixed Points')
    axes[1].set_title('Transformed Moving Points')
    axes[1].legend()
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.show()


def align_with_cellpose(moving_img, fixed_img, scale_factor=1, verbose=False, show=False, **kwargs):
    if verbose:
        print('Getting moving masks...', end='')
    moving_masks = get_cellpose_masks(moving_img)
    if verbose:
        print(f'done, found {moving_masks.max()} cells in moving image.')
        print('Getting fixed masks...', end='')
    fixed_masks = get_cellpose_masks(fixed_img)

    if verbose:
        print(f'done, found {fixed_masks.max()} cells in fixed image.')
        print('Registering centroids...')
    transform = masks_to_alignment(moving_masks, fixed_masks, scale_factor=scale_factor, show=show, verbose=verbose, **kwargs)

    return transform

def find_initial_translation(p_moving, p_fixed, bin_size: int = 10):
    """
    Finds the optimal initial translation for two 2D point clouds using
    2D histogram cross-correlation. Assumes rotation is minimal!!

    Args:
        p_moving (np.ndarray): moving point cloud of shape (M, 2).
        p_fixed (np.ndarray): fixed point cloud of shape (N, 2).
        bin_size (float): The size of each bin in the 2D histogram.

    Returns:
        np.ndarray: The optimal translation vector of shape (2,).
    """
    # 1. Define the grid
    # Find the overall bounds for both clouds combined
    all_points = np.vstack([p_moving, p_fixed])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)

    # Define the histogram bins based on the bounds and bin_size
    x_bins = np.arange(min_coords[0], max_coords[0] + bin_size, bin_size)
    y_bins = np.arange(min_coords[1], max_coords[1] + bin_size, bin_size)
    bins = [x_bins, y_bins]

    # 2. Create Density Maps
    hist_moving, _, _ = np.histogram2d(p_moving[:, 0], p_moving[:, 1], bins=bins)
    hist_fixed, _, _ = np.histogram2d(p_fixed[:, 0], p_fixed[:, 1], bins=bins)

    # 3. Compute Cross-Correlation
    # We use 'full' mode to explore all possible overlaps
    correlation_map = correlate2d(hist_fixed, hist_moving, mode='full')

    # 4. Find the Peak and Optimal Translation
    # Find the index of the peak in the correlation map
    peak_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    peak_idx = np.array(peak_idx)

    # The center of the correlation map corresponds to zero shift.
    # We need to find the offset of the peak from the center.
    center_idx = np.array(hist_moving.shape) - 1

    # The shift is the difference between the peak and the center
    # Note: Scipy's correlation flips the second kernel, so we use hist_moving shape
    shift_bins = peak_idx - center_idx

    # Convert shift from bin units to physical units
    translation_vector = shift_bins * bin_size

    return translation_vector

def compute_ICP(p_moving, p_fixed, initial_transform=SimilarityTransform(), max_iterations=50, tolerance=1e-6, verbose=False):
    """
    Refines a transformation using the Iterative Closest Point (ICP) algorithm.

    Args:
        p_moving (np.ndarray): The original moving point cloud (M, 2).
        p_fixed (np.ndarray): The fixed point cloud (N, 2).
        initial_transform (SimilarityTransform): An initial guess for the transform.
        max_iterations (int): The maximum number of ICP iterations.
        tolerance (float): Convergence criteria for change in the transform matrix.

    Returns:
        SimilarityTransform: The final, refined transformation.
    """
    # 2. Build KD-Tree on the fixed cloud for fast lookups
    kdtree_fixed = cKDTree(p_fixed)

    cumulative_transform = initial_transform
    prev_transform_matrix = cumulative_transform.params

    for i in range(max_iterations):
        # 3a. Transform the moving points with the current best estimate
        p_transformed_moving = cumulative_transform(p_moving)

        # 3b. Find the closest neighbors in the fixed cloud
        distances, indices = kdtree_fixed.query(p_transformed_moving)

        # 3c. Filter out matches that are too far away
        cutoff_distance = np.median(distances) * 2  # Set a cutoff distance based on median distance
        valid_matches = distances < cutoff_distance
        mean_distance = np.mean(distances[valid_matches])

        # These are our corresponding point sets
        src_matches = p_transformed_moving[valid_matches]
        dst_matches = p_fixed[indices[valid_matches]]

        # 3d. Estimate the incremental transform to fix the remaining error
        # NOTE: We use the *transformed* moving points to find the *incremental* transform
        incremental_transform = estimate_transform('similarity', src_matches, dst_matches)

        # 3e. Update the cumulative transformation
        # The new total transform is the new small step applied AFTER the previous total transform.
        # However, it's easier to compose them like this: T_new(p) = T_inc(T_old(p))
        # Matrix composition: M_new = M_inc @ M_old
        cumulative_transform = SimilarityTransform(matrix=(incremental_transform.params @ cumulative_transform.params))

        # 3f. Check for convergence
        change = np.max(np.abs(cumulative_transform.params - prev_transform_matrix))
        if verbose:
            print(
                f'Iteration {i}: Mean correspondence distance = {mean_distance:.4f}, Matching {len(src_matches)} points, Transform change = {change:.4g}'
            )
        if change < tolerance:
            if verbose:
                print('Converged.')
                if mean_distance > 10:
                    print('Warning: Mean correspondence distance is high, indicating potential issues with the match quality.')
            break

        prev_transform_matrix = cumulative_transform.params

    return cumulative_transform
