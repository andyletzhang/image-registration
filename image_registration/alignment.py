from typing import Literal, Tuple, Union

import cv2
import numpy as np
from . import registration
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_erosion


class Aligner:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        verbose=False,
        transformation: tuple = None,
    ) -> None:
        self.verbose = verbose

        source_shape = source.shape
        self.source, source_dims = reshape_image_array(source)
        self.source_TPZ = source_shape[: source_dims.index("Y")]
        self.source_YX = self.source.shape[-3:-1]

        target_shape = target.shape
        self.target, target_dims = reshape_image_array(target)
        self.target_TPZ = target_shape[: target_dims.index("Y")]
        self.target_YX = self.target.shape[-3:-1]

        # Reshape to (N, Y, X, C)
        self.source = self.source.reshape(-1, *self.source.shape[-3:])
        self.target = self.target.reshape(-1, *self.target.shape[-3:])

        if self.source.shape[0] != self.target.shape[0]:
            raise ValueError(
                f"Source and target stacks must have the same number of slices: found {self.source.shape[0]} and {self.target.shape[0]}."
            )

    def compute_registration(self, idx=0, show=False):
        out = registration.align_with_cellpose(
            self.source[idx], self.target[idx], verbose=self.verbose, show=show
        )
        self.scale, self.rotation, self.translation = out["scale"], -out["rotation_angle"], out["translation"]
        if self.verbose:
            print("Alignment transformation computed:")
            print(f"scale = {self.scale}")
            print(f"rotation = {self.rotation}")
            print(f"translation = {self.translation}")

        return self.scale, self.rotation, self.translation

    def keypoint_registration(self, src_keypoints: np.ndarray, tgt_keypoints: np.ndarray):
        if src_keypoints.shape != tgt_keypoints.shape:
            raise ValueError("Keypoints must have the same shape.")
        self.scale, self.rotation, self.translation = registration.compute_similarity_transform(src_keypoints, tgt_keypoints)
        if self.verbose:
            print("Alignment transformation computed:")
            print(f"scale = {self.scale}")
            print(f"rotation = {self.rotation}")
            print(f"translation = {self.translation}")

        return self.scale, self.rotation, self.translation

    def align(
        self,
        method="full",
        pad_value=0,
        interpolation=cv2.INTER_LINEAR,
        show_progress=False,
        transformation=None,
    ):
        if transformation:
            self.scale, self.rotation, self.translation = transformation

        if not hasattr(self, "scale"):
            if self.verbose:
                print("Computing registration...")
            self.compute_registration()

        self.source_aligned, self.target_aligned = self._align_stacks(
            self.source,
            self.target,
            pad_value=pad_value,
            method=method,
            interpolation=interpolation,
            show_progress=show_progress,
        )

        self.source_aligned = self.source_aligned.reshape(
            *self.target_TPZ, *self.source_aligned.shape[-3:]
        )
        self.target_aligned = self.target_aligned.reshape(
            *self.target_TPZ, *self.target_aligned.shape[-3:]
        )

        return self.source_aligned, self.target_aligned

    @property
    def M(self):
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = self.scale * self.R
        M[:, 2] = self.translation - self.offset

        return M

    @property
    def R(self):
        return degrees_to_rotation_matrix(self.rotation)

    def compute_size_offset(self):
        # Compute transformed source image corners (using first slice dimensions)
        h_src, w_src = self.source_YX
        corners = np.array([[0, 0], [w_src, 0], [0, h_src], [w_src, h_src]])
        transformed_corners = self.scale * (corners @ self.R.T) + self.translation

        # Compute final canvas bounding box
        min_x = min(0, np.min(transformed_corners[:, 0]))
        min_y = min(0, np.min(transformed_corners[:, 1]))
        max_x = max(self.target_YX[0], np.max(transformed_corners[:, 0]))
        max_y = max(self.target_YX[1], np.max(transformed_corners[:, 1]))

        # Compute final output size
        self._output_size = (int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x)))

        # Compute offset for translation correction
        self._offset = np.array([min_x, min_y])

    @property
    def output_size(self):
        if not hasattr(self, "_output_size"):
            self.compute_size_offset()
        return self._output_size

    @property
    def offset(self):
        if not hasattr(self, "_offset"):
            self.compute_size_offset()
        return self._offset

    def _align_stacks(
        self,
        source_stack: np.ndarray,
        target_stack: np.ndarray,
        pad_value: Union[int, float] = 0,
        method: Literal["full", "crop"] = "full",
        interpolation: int = cv2.INTER_LINEAR,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Aligns source stack to target stack using the provided transformation.
        Efficiently processes the entire stack at once with the same transformation.

        Args:
            source_stack: Source image stack with shape (n, y, x, c)
            target_stack: Target image stack with shape (n, y, x, c)
            transformation: Tuple of (scale, rotation, translation)
                - scale: float scaling factor
                - rotation: float angle in degrees or 2x2 rotation matrix
                - translation: 2D translation vector [tx, ty]
            pad_value: Value used for padding
            method: 'full' returns the full aligned image, 'crop' crops to the valid overlap region
            interpolation: Interpolation method for warping

        Returns:
            Merged z-stack containing the aligned source and target
        """
        if source_stack.ndim != 4 or target_stack.ndim != 4:
            raise ValueError("Input stacks must have shape (n, y, x, c).")

        if self.verbose:
            print("Performing alignment transformations...")
        # Apply transformation to each z-slice
        source_transformed = self.transform_source(source_stack, pad_value=pad_value, interpolation=interpolation, show_progress=show_progress)

        # Pad target image to match final size
        target_padded = self.transform_target(target_stack, pad_value=pad_value)

        if self.verbose:
            print("Alignment complete!")
        if method == "full":
            # Return all image data
            return source_transformed, target_padded
        elif method == "crop":
            if self.verbose:
                print("Cropping to valid region...")
            # Identify valid region for cropping
            x1, y1, x2, y2 = self.valid_bounds
            
            return source_transformed[..., y1:y2, x1:x2, :], target_padded[..., y1:y2, x1:x2, :]
        else:
            raise ValueError(f"Unknown method: {method}. Expected 'full' or 'crop'.")

    def get_valid_region(self):
        src_mask = np.ones((1, *self.source_YX, 1), dtype=np.uint8)
        src_mask = self.transform_source(src_mask, pad_value=0)
        src_mask = src_mask > 0

        target_mask = np.ones((1, *self.target_YX, 1), dtype=np.uint8)
        target_mask = self.transform_target(target_mask, pad_value=0)

        valid_region = np.logical_and(src_mask, target_mask)
        self._valid_bounds = get_largest_valid_rectangle(valid_region[0,...,0])
        
        return self._valid_bounds

    @property
    def valid_bounds(self):
        if not hasattr(self, "_valid_bounds"):
            self.get_valid_region()
        return self._valid_bounds

    def transform_source(self, source_stack, pad_value=0, interpolation=cv2.INTER_LINEAR, show_progress=False):
        if source_stack.ndim != 4:
            source_stack=reshape_image_array(source_stack)[0]
            source_stack=source_stack.reshape(-1, *source_stack.shape[-3:])
        
        slices, n_channels = source_stack.shape[0], source_stack.shape[3]
        source_transformed_shape = (slices, *self.output_size, n_channels)
        source_transformed = np.zeros(source_transformed_shape, dtype=np.float32)

        if show_progress:
            from tqdm.auto import tqdm
            progress = tqdm(desc="Transforming source images", total=slices*n_channels)

        for z in range(slices):
            for c in range(n_channels):
                # Apply transformation
                source_warped = cv2.warpAffine(
                    source_stack[z, :, :, c],
                    self.M,
                    self.output_size[::-1],
                    flags=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=pad_value
                )
                if not np.allclose(self.R, np.eye(2)):
                    source_warped = remove_edge_pixels(source_warped, pad_value)
                source_transformed[z, :, :, c] = source_warped

                if show_progress:
                    progress.update(1)

        if show_progress:
            progress.close()
        
        return source_transformed

    def transform_target(self, target_stack, pad_value=0):
        if target_stack.ndim != 4:
            target_stack=reshape_image_array(target_stack)[0]
            target_stack=target_stack.reshape(-1, *target_stack.shape[-3:])

        slices, n_channels=target_stack.shape[0], target_stack.shape[3]
        output_h, output_w = self.output_size

        target_shape = (slices, output_h, output_w, n_channels)
        if np.isscalar(pad_value):
            target_padded = np.full(target_shape, pad_value, dtype=target_stack.dtype)
        else:
            raise ValueError(f"Unexpected pad value {pad_value}: Only scalar pad values are supported.")

        # Place the target image into the padded canvas
        min_x, min_y = self.offset
        y_start = max(0, -int(min_y))
        y_end = min(output_h, target_stack.shape[1] - int(min_y))
        x_start = max(0, -int(min_x))
        x_end = min(output_w, target_stack.shape[2] - int(min_x))

        target_y_start = max(0, int(min_y))
        target_y_end = target_y_start + (y_end - y_start)
        target_x_start = max(0, int(min_x))
        target_x_end = target_x_start + (x_end - x_start)

        target_padded[:, y_start:y_end, x_start:x_end, :] = target_stack[
            :, target_y_start:target_y_end, target_x_start:target_x_end, :
        ]

        return target_padded

def align_imgs(
    source: np.ndarray,
    target: np.ndarray,
    method: str = "full",
    pad_value=0,
    transformation: tuple = None,
    interpolation=cv2.INTER_LINEAR,
    verbose: bool = True,
    show_alignment: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two image stacks.
    """

    aligner = Aligner(source, target, verbose=verbose)
    if not transformation:
        transformation = aligner.compute_registration(show=show_alignment)
    return aligner.align(
        method=method,
        pad_value=pad_value,
        interpolation=interpolation,
        transformation=transformation,
    )


def reshape_image_array(image: np.ndarray, axes: str = None):
    axis_order = "TPZYXC"
    shape = [1] * 6

    if image.ndim < 2 or image.ndim > 6:
        raise ValueError(
            f"Unexpected image shape {image.shape}. Image must have between 2 and 5 dimensions"
        )

    if axes:  # reshape the image using the provided axis order
        if len(axes) != image.ndim:
            raise ValueError("Axes must have the same length as the image dimensions")

        axes = axes.upper()
        for i, ax in enumerate(axes):
            if ax not in axis_order:
                raise ValueError(f"Invalid axis {ax}. Must be one of {axis_order}")
            shape[axis_order.index(ax)] = image.shape[i]

    else:  # infer the axis order from the image shape (to the best of our ability)
        current_shape = image.shape
        if len(current_shape) == 2:  # grayscale, 2D image
            shape[3] = current_shape[0]
            shape[4] = current_shape[1]
            axes = "YX"

        elif len(current_shape) == 6:  # all axes accounted for
            shape = current_shape
            axes = axis_order

        else:
            if (
                current_shape[-1] > 6
            ):  # assume the last dimension is X (large size) instead of channels (small size)
                axes = "TPZYX"[-len(current_shape) :]
                current_shape = current_shape + (1,)
            else:
                axes = "TPZYXC"[-len(current_shape) :]
            shape = (1,) * (6 - len(current_shape)) + current_shape

    return image.reshape(shape), axes


def transformation_matrix(
    scale: float, R: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Constructs 2x3 affine transformation matrix.

    Args:
        scale: Scaling factor for the transformation
        R: 2x2 rotation matrix
        translation: 2D translation vector [tx, ty]

    Returns:
        2x3 affine transformation matrix
    """
    return np.hstack([scale * R, translation.reshape(2, 1)])


def degrees_to_rotation_matrix(degrees: float) -> np.ndarray:
    """Converts degrees to a 2D rotation matrix.

    Args:
        degrees: Rotation angle in degrees

    Returns:
        2x2 rotation matrix
    """
    theta = degrees * np.pi / 180
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def remove_edge_pixels(
    image: np.ndarray, pad_value: Union[int, float, np.ndarray]
) -> np.ndarray:
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
    all_pairs = np.concatenate(
        [np.repeat(v1, v2.shape[1], axis=1), np.repeat(v2, v1.shape[0], axis=0)], axis=2
    ).reshape(-1, 4)
    # sort x1 x2, y1 y2
    all_pairs[:, [0, 2]] = np.sort(all_pairs[:, [0, 2]], axis=1)
    all_pairs[:, [1, 3]] = np.sort(all_pairs[:, [1, 3]], axis=1)
    areas = (all_pairs[:, 2] - all_pairs[:, 0]) * (all_pairs[:, 3] - all_pairs[:, 1])
    idx = np.argsort(areas)[::-1]

    for y1, x1, y2, x2 in all_pairs[idx]:
        if np.all(valid[y1:y2, x1:x2]):
            return x1, y1, x2, y2
    else:
        raise(ValueError("No valid bounds found."))