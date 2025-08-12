from pathlib import Path
import yaml
import tifffile


class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)
            # Use the first base_path, first camera, and image_format for dtype detection
            source_path = Path(self.source_paths[0])
            camera_folder = f"Cam{self.camera_numbers[0]}"
            # Use correct image format for dtype detection
            if self.time_resolved:
                file_path = source_path / camera_folder / (self.image_format % 1)
            else:
                file_path = source_path / camera_folder / (self.image_format[0] % 1)
            img = tifffile.imread(file_path)
            self.image_dtype = img.dtype

    @property
    def time_resolved(self):
        return self.data['images'].get('time_resolved', False)

    @property
    def image_format(self):
        if self.time_resolved:
            return self.data['images'].get('image_format', "B%05d.tiff")
        else:
            # Expect a list of two formats in the config for A and B images
            fmts = self.data['images'].get('image_format', ["B%05d_A.tiff", "B%05d_B.tiff"])
            return tuple(fmts)

    @property
    def base_paths(self):
        return [Path(p) for p in self.data['paths']['base_paths']]

    @property
    def source_paths(self):
        return [Path(s) for s in self.data['paths']['source_paths']]

    @property
    def camera_numbers(self):
        return self.data['paths']['camera_numbers']

    @property
    def camera_folders(self):
        return [f"Cam{n}" for n in self.camera_numbers]

    @property
    def num_images(self):
        return self.data['images']['num_images']

    @property
    def image_shape(self):
        return tuple(self.data['images']['shape'])

    @property
    def piv_chunk_size(self):
        # Updated to use batches.size from config.yaml
        return self.data['batches']['size']

    @property
    def filter_type(self):
        # This is now optional, as filters block is used
        return self.data.get('pre_procesing', {}).get('filter_type', None)

    @property
    def filters(self):
        # Returns the list of filter dicts from config.yaml
        return self.data.get('filters', [])

    @property
    def vector_format(self):
        # Returns a single format string like "B%05d.mat"
        vf = self.data['images'].get('vector_format', ["B%05d.mat"])
        if isinstance(vf, (list, tuple)):
            return vf[0]
        return vf

    @property
    def statistics_extraction(self):
        # Returns the statistics_extraction block as a list, or empty list if not present
        return self.data.get('statistics_extraction', [])

    @property
    def instantaneous_runs(self):
        return self.data.get('instantaneous_piv', {}).get('runs', [])

    @property
    def instantaneous_window_sizes(self):
        return self.data.get('instantaneous_piv', {}).get('window_size', [])

    @property
    def instantaneous_overlaps(self):
        return self.data.get('instantaneous_piv', {}).get('overlap', [])

    @property
    def plots(self):
        return self.data.get('plots', [])

    @property
    def videos(self):
        """
        Returns the 'videos' list from config.yaml. Ensures a list is returned.
        Each entry may contain: type, endpoint, use_merged, video_length, variable.
        """
        vids = self.data.get('videos', [])
        if vids is None:
            return []
        if isinstance(vids, dict):
            return [vids]
        return list(vids)
