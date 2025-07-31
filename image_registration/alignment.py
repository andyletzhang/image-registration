from typing import Literal

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_erosion
from skimage.transform import SimilarityTransform, estimate_transform, warp

from . import registration


class Aligner:
    def __init__(self, moving: np.ndarray, fixed: np.ndarray, verbose=False, transform: SimilarityTransform = None, scale_factor: float = 1.0) -> None:
        self.verbose = verbose
        self.transform = transform
        self.scale_factor = scale_factor

        moving_shape = moving.shape
        self.moving, moving_dims = reshape_image_array(moving)
        self.moving_TPZ = moving_shape[: moving_dims.index('Y')]
        self.moving_YX = self.moving.shape[-3:-1]

        fixed_shape = fixed.shape
        self.fixed, fixed_dims = reshape_image_array(fixed)
        self.fixed_TPZ = fixed_shape[: fixed_dims.index('Y')]
        self.fixed_YX = self.fixed.shape[-3:-1]

        # Reshape to (N, Y, X, C)
        self.moving = self.moving.reshape(-1, *self.moving.shape[-3:])
        self.fixed = self.fixed.reshape(-1, *self.fixed.shape[-3:])

        if self.moving.shape[0] != self.fixed.shape[0]:
            raise ValueError(
                f'moving and fixed stacks must have the same number of slices: found {self.moving.shape[0]} and {self.fixed.shape[0]}.'
            )

    def compute_registration(self, idx=0, show=False):
        self.transform = registration.align_with_cellpose(self.moving[idx], self.fixed[idx], verbose=self.verbose, show=show, scale_factor=self.scale_factor)
        if self.verbose:
            print('Alignment transformation computed:')
            print(f'scale = {self.transform.scale}')
            print(f'rotation = {self.transform.rotation}')
            print(f'translation = {self.transform.translation}')

        return self.transform

    def keypoint_registration(self, mvg_keypoints: np.ndarray, fxd_keypoints: np.ndarray):
        if mvg_keypoints.shape != fxd_keypoints.shape:
            raise ValueError('Keypoints must have the same shape.')
        self.transform = estimate_transform(
            'similarity',
            mvg_keypoints,
            fxd_keypoints,
        )
        if self.verbose:
            print('Alignment transformation computed:')
            print(f'scale = {self.transform.scale}')
            print(f'rotation = {self.transform.rotation}')
            print(f'translation = {self.transform.translation}')

        return self.transform

    def align(self, method='full', pad_value=0, show_progress=False, transform: SimilarityTransform = None):
        if transform:
            self.transform = transform

        if not hasattr(self, 'transform') or self.transform is None:
            if self.verbose:
                print('Computing registration...')
            self.compute_registration()

        self.moving_aligned, self.fixed_aligned = self._align_stacks(
            self.moving, self.fixed, pad_value=pad_value, method=method, show_progress=show_progress
        )

        self.moving_aligned = self.moving_aligned.reshape(*self.fixed_TPZ, *self.moving_aligned.shape[-3:])
        self.fixed_aligned = self.fixed_aligned.reshape(*self.fixed_TPZ, *self.fixed_aligned.shape[-3:])

        return self.moving_aligned, self.fixed_aligned

    @property
    def M(self):
        transform = SimilarityTransform(
            scale=self.transform.scale,
            rotation=self.transform.rotation,
            translation=self.transform.translation-self.offset,
        )
        return transform.params[:2, :3]

    @property
    def R(self):
        return rad_to_rotation_matrix(self.transform.rotation)

    def compute_size_offset(self):
        # Compute transformed moving image corners (using first slice dimensions)
        h_mvg, w_mvg = self.moving_YX
        corners = np.array([[0, 0], [w_mvg, 0], [0, h_mvg], [w_mvg, h_mvg]])
        transformed_corners = self.transform(corners)

        # Compute final canvas bounding box
        min_x = min(0, np.min(transformed_corners[:, 0]))
        min_y = min(0, np.min(transformed_corners[:, 1]))
        max_x = max(self.fixed_YX[0], np.max(transformed_corners[:, 0]))
        max_y = max(self.fixed_YX[1], np.max(transformed_corners[:, 1]))

        # Compute final output size
        self._output_size = (int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x)))

        # Compute offset for translation correction
        self._offset = np.array([min_x, min_y])

    @property
    def output_size(self):
        if not hasattr(self, '_output_size'):
            self.compute_size_offset()
        return self._output_size

    @property
    def offset(self):
        if not hasattr(self, '_offset'):
            self.compute_size_offset()
        return self._offset

    def _align_stacks(
        self,
        moving_stack: np.ndarray,
        fixed_stack: np.ndarray,
        pad_value: int | float = 0,
        method: Literal['full', 'crop'] = 'full',
        show_progress: bool = False,
    ) -> np.ndarray:
        """Aligns moving stack to fixed stack using the provided transformation.
        Efficiently processes the entire stack at once with the same transformation.

        Args:
            moving_stack: moving image stack with shape (n, y, x, c)
            fixed_stack: fixed image stack with shape (n, y, x, c)
            transformation: Tuple of (scale, rotation, translation)
                - scale: float scaling factor
                - rotation: float angle in degrees or 2x2 rotation matrix
                - translation: 2D translation vector [tx, ty]
            pad_value: Value used for padding
            method: 'full' returns the full aligned image, 'crop' crops to the valid overlap region
            interpolation: Interpolation method for warping

        Returns:
            Merged z-stack containing the aligned moving and fixed
        """
        if moving_stack.ndim != 4 or fixed_stack.ndim != 4:
            raise ValueError('Input stacks must have shape (n, y, x, c).')

        if self.verbose:
            print('Performing alignment transformations...')
        # Apply transformation to each z-slice
        moving_transformed = self.transform_moving(
            moving_stack, pad_value=pad_value, show_progress=show_progress
        )

        # Pad fixed image to match final size
        fixed_padded = self.transform_fixed(fixed_stack, pad_value=pad_value)

        if self.verbose:
            print('Alignment complete!')
        if method == 'full':
            # Return all image data
            return moving_transformed, fixed_padded
        elif method == 'crop':
            if self.verbose:
                print('Cropping to valid region...')
            # Identify valid region for cropping
            x1, y1, x2, y2 = self.valid_bounds

            return moving_transformed[..., y1:y2, x1:x2, :], fixed_padded[..., y1:y2, x1:x2, :]
        else:
            raise ValueError(f"Unknown method: {method}. Expected 'full' or 'crop'.")

    def get_valid_region(self):
        mvg_mask = np.ones((1, *self.moving_YX, 1), dtype=np.uint8)
        mvg_mask = self.transform_moving(mvg_mask, pad_value=0)
        mvg_mask = mvg_mask > 0

        fixed_mask = np.ones((1, *self.fixed_YX, 1), dtype=np.uint8)
        fixed_mask = self.transform_fixed(fixed_mask, pad_value=0)

        valid_region = np.logical_and(mvg_mask, fixed_mask)
        self._valid_bounds = get_largest_valid_rectangle(valid_region[0, ..., 0])

        return self._valid_bounds

    @property
    def valid_bounds(self):
        if not hasattr(self, '_valid_bounds'):
            self.get_valid_region()
        return self._valid_bounds

    def transform_moving(self, moving_stack, pad_value=0, show_progress=False):
        if moving_stack.ndim != 4:
            moving_stack = reshape_image_array(moving_stack)[0]
            moving_stack = moving_stack.reshape(-1, *moving_stack.shape[-3:])

        slices, n_channels = moving_stack.shape[0], moving_stack.shape[3]
        moving_transformed_shape = (slices, *self.output_size, n_channels)
        moving_transformed = np.zeros(moving_transformed_shape, dtype=np.float32)

        if show_progress:
            from tqdm.auto import tqdm

            progress = tqdm(desc='Transforming moving images', total=slices * n_channels)

        img_transform = SimilarityTransform(
            rotation=self.transform.rotation,
            scale=self.transform.scale,
            translation=self.transform.translation - self.offset
        )
        for z in range(slices):
            for c in range(n_channels):
                # Apply transformation
                moving_warped = warp(moving_stack[z, :, :, c], img_transform.inverse, output_shape=self.output_size)
                if not np.allclose(self.R, np.eye(2)):
                    moving_warped = remove_edge_pixels(moving_warped, pad_value)
                moving_transformed[z, :, :, c] = moving_warped

                if show_progress:
                    progress.update(1)

        if show_progress:
            progress.close()

        return moving_transformed

    def transform_fixed(self, fixed_stack, pad_value=0):
        if fixed_stack.ndim != 4:
            fixed_stack = reshape_image_array(fixed_stack)[0]
            fixed_stack = fixed_stack.reshape(-1, *fixed_stack.shape[-3:])

        slices, n_channels = fixed_stack.shape[0], fixed_stack.shape[3]
        output_h, output_w = self.output_size

        fixed_shape = (slices, output_h, output_w, n_channels)
        if np.isscalar(pad_value):
            fixed_padded = np.full(fixed_shape, pad_value, dtype=fixed_stack.dtype)
        else:
            raise ValueError(f'Unexpected pad value {pad_value}: Only scalar pad values are supported.')

        # Place the fixed image into the padded canvas
        min_x, min_y = self.offset
        y_start = max(0, -int(min_y))
        y_end = min(output_h, fixed_stack.shape[1] - int(min_y))
        x_start = max(0, -int(min_x))
        x_end = min(output_w, fixed_stack.shape[2] - int(min_x))

        fixed_y_start = max(0, int(min_y))
        fixed_y_end = fixed_y_start + (y_end - y_start)
        fixed_x_start = max(0, int(min_x))
        fixed_x_end = fixed_x_start + (x_end - x_start)

        fixed_padded[:, y_start:y_end, x_start:x_end, :] = fixed_stack[:, fixed_y_start:fixed_y_end, fixed_x_start:fixed_x_end, :]

        return fixed_padded


def align_imgs(
    moving: np.ndarray,
    fixed: np.ndarray,
    method: str = 'full',
    pad_value=0,
    transformation: tuple = None,
    verbose: bool = True,
    show_alignment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two image stacks.
    """

    aligner = Aligner(moving, fixed, verbose=verbose)
    if not transformation:
        transformation = aligner.compute_registration(show=show_alignment)
    return aligner.align(method=method, pad_value=pad_value, transformation=transformation)


def reshape_image_array(image: np.ndarray, axes: str = None):
    axis_order = 'TPZYXC'
    shape = [1] * 6

    if image.ndim < 2 or image.ndim > 6:
        raise ValueError(f'Unexpected image shape {image.shape}. Image must have between 2 and 5 dimensions')

    if axes:  # reshape the image using the provided axis order
        if len(axes) != image.ndim:
            raise ValueError('Axes must have the same length as the image dimensions')

        axes = axes.upper()
        for i, ax in enumerate(axes):
            if ax not in axis_order:
                raise ValueError(f'Invalid axis {ax}. Must be one of {axis_order}')
            shape[axis_order.index(ax)] = image.shape[i]

    else:  # infer the axis order from the image shape (to the best of our ability)
        current_shape = image.shape
        if len(current_shape) == 2:  # grayscale, 2D image
            shape[3] = current_shape[0]
            shape[4] = current_shape[1]
            axes = 'YX'

        elif len(current_shape) == 6:  # all axes accounted for
            shape = current_shape
            axes = axis_order

        else:
            if current_shape[-1] > 6:  # assume the last dimension is X (large size) instead of channels (small size)
                axes = 'TPZYX'[-len(current_shape) :]
                current_shape = current_shape + (1,)
            else:
                axes = 'TPZYXC'[-len(current_shape) :]
            shape = (1,) * (6 - len(current_shape)) + current_shape

    return image.reshape(shape), axes


def transformation_matrix(scale: float, R: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Constructs 2x3 affine transformation matrix.

    Args:
        scale: Scaling factor for the transformation
        R: 2x2 rotation matrix
        translation: 2D translation vector [tx, ty]

    Returns:
        2x3 affine transformation matrix
    """
    return np.hstack([scale * R, translation.reshape(2, 1)])


def rad_to_rotation_matrix(radians: float) -> np.ndarray:
    """Converts radians to a 2D rotation matrix.

    Args:
        degrees: Rotation angle in degrees

    Returns:
        2x2 rotation matrix
    """
    return np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])


def remove_edge_pixels(image: np.ndarray, pad_value: int | float | np.ndarray) -> np.ndarray:
    """Removes immediate boundary pixels which may be quantitatively affected by rotation.

    Args:
        image: Input image of any dimensionality
        pad_value: Value used for padding

    Returns:
        Image with edge pixels removed
    """
    # Create a working copy to avoid modifying the original
    result = image.copy()

    # Handle pad_value for multi-channel images
    if isinstance(pad_value, (list, tuple, np.ndarray)) and image.ndim > 2:
        binarized = np.all(image != pad_value, axis=-1)
    else:
        binarized = image != pad_value

    if binarized.ndim == 3:
        binarized = np.any(binarized, axis=2)

    # Fill holes
    binarized = binary_fill_holes(binarized)

    # Erode twice to get rid of boundary pixels
    eroded = binary_erosion(binary_erosion(binarized))

    # Create border mask
    border = binarized & ~eroded

    # Apply border mask to all channels
    if image.ndim == 3:
        border = np.expand_dims(border, axis=2)
        border = np.repeat(border, image.shape[2], axis=2)

    result[border] = pad_value
    return result


# Crop to Data Functions
def get_largest_valid_rectangle(valid):
    boundary = binary_erosion(valid) ^ valid

    # get all vertex pairs on opposite sides of the boundary
    vertices = np.argwhere(boundary)
    v1 = vertices[vertices[:, 0] >= boundary.shape[0] // 2][:, np.newaxis, :]
    v2 = vertices[vertices[:, 0] < boundary.shape[0] // 2][np.newaxis, :, :]

    # Get all pairs in a single array with shape (N, M, 4)
    # Each pair contains [x1, y1, x2, y2]
    all_pairs = np.concatenate([np.repeat(v1, v2.shape[1], axis=1), np.repeat(v2, v1.shape[0], axis=0)], axis=2).reshape(-1, 4)
    # sort x1 x2, y1 y2
    all_pairs[:, [0, 2]] = np.sort(all_pairs[:, [0, 2]], axis=1)
    all_pairs[:, [1, 3]] = np.sort(all_pairs[:, [1, 3]], axis=1)
    areas = (all_pairs[:, 2] - all_pairs[:, 0]) * (all_pairs[:, 3] - all_pairs[:, 1])
    idx = np.argsort(areas)[::-1]

    for y1, x1, y2, x2 in all_pairs[idx]:
        if np.all(valid[y1:y2, x1:x2]):
            return x1, y1, x2, y2
    else:
        raise (ValueError('No valid bounds found.'))
