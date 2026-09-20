import numpy as np
from scipy import ndimage


class SliceAugmenter:
    

    def __init__(self, seed=42,
                 rotation_degrees=15.0,
                 scale_range=(0.9, 1.1),
                 flip_prob=0.5,
                 brightness=0.06,
                 contrast_range=(0.9, 1.1),
                 noise_std=0.01,
                 geometric_prob=0.5,
                 intensity_prob=0.5,
                 intensity_channels=None):
        self.seed = seed
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.geometric_prob = geometric_prob
        self.intensity_prob = intensity_prob
        self.intensity_channels = intensity_channels
        self._rng = None

    def _ensure_rng(self):
        
        if self._rng is None:
            try:
                import torch.utils.data
                info = torch.utils.data.get_worker_info()
                worker_id = info.id if info is not None else 0
            except Exception:
                worker_id = 0
            self._rng = np.random.RandomState(self.seed + 1013 * worker_id)

    def _geometric(self, image, mask):
        rng = self._rng
        angle = np.deg2rad(rng.uniform(-self.rotation_degrees, self.rotation_degrees))
        scale = rng.uniform(*self.scale_range)

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Maps output coordinates back to input coordinates, so this single
        # matrix does the rotation and the zoom together.
        matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64) / scale
        centre = (np.array(mask.shape, dtype=np.float64) - 1) / 2.0
        offset = centre - matrix @ centre

        warped = np.empty_like(image)
        for c in range(image.shape[0]):
            warped[c] = ndimage.affine_transform(
                image[c], matrix, offset=offset, order=1, mode="constant", cval=0.0)

        warped_mask = ndimage.affine_transform(
            mask.astype(np.float32), matrix, offset=offset,
            order=0, mode="constant", cval=0.0).astype(mask.dtype)

        return warped, warped_mask

    def _intensity(self, image):
        rng = self._rng
        channels = self.intensity_channels
        if channels is None:
            channels = range(image.shape[0])

        contrast = rng.uniform(*self.contrast_range)
        shift = rng.uniform(-self.brightness, self.brightness)

        image = image.copy()
        for c in channels:
            channel = image[c] * contrast + shift
            if self.noise_std > 0:
                channel = channel + rng.normal(0.0, self.noise_std, size=channel.shape)
            image[c] = np.clip(channel, 0.0, 1.0)
        return image

    def __call__(self, image, mask):
        self._ensure_rng()
        rng = self._rng

        if rng.rand() < self.flip_prob:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        if rng.rand() < self.geometric_prob:
            image, mask = self._geometric(image, mask)

        if rng.rand() < self.intensity_prob:
            image = self._intensity(image)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
