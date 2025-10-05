import logging
from pathlib import Path

import tifffile
import yaml


class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)
            file_path = self.base_path / self.cameras[0] / (self.image_format % 1)
            img = tifffile.imread(file_path)
            self.image_dtype = img.dtype

        self._setup_logging()

        if self.data["piv"]["type"].lower() == "instantaneous":
            self.ensemble = False
        elif self.data["piv"]["type"].lower() == "ensemble":
            self.ensemble = True
        else:
            raise ValueError("piv.type must be 'instantaneous' or 'ensemble'")

        logging.info("Config loaded successfully")
        logging.debug("Image dtype: %s", self.image_dtype)

    @property
    def image_format(self):
        return self.data["images"]["image_format"]

    @property
    def base_path(self):
        return Path(self.data["paths"]["base_path"])

    @property
    def cameras(self):
        return self.data["paths"]["camera_folders"]

    @property
    def num_images(self):
        return self.data["images"]["num_images"]

    @property
    def image_shape(self):
        return tuple(self.data["images"]["shape"])

    @property
    def piv_chunk_size(self):
        return self.data["piv"]["chunk_size"]

    @property
    def filters(self):
        return self.data["preprocessing"]["filters"]

    @property
    def piv_type(self):
        return self.data["piv"]["type"]

    @property
    def window_sizes(self):
        return self.data["piv"]["window_size"]

    @property
    def overlap(self):
        return self.data["piv"]["overlap"]

    @property
    def num_peaks(self):
        return self.data["piv"]["num_peaks"]

    @property
    def dt(self):
        return self.data["piv"]["dt"]

    @property
    def window_type(self):
        return self.data.get("piv", {}).get("window_type", "A")

    @property
    def backend(self):
        return self.data.get("backend", "cpu").lower()

    @property
    def num_passes(self):
        return len(self.window_sizes)

    @property
    def debug(self):
        return self.data.get("processing", {}).get("debug", False)

    @property
    def omp_threads(self):
        return str(self.data.get("processing", {}).get("omp_threads", 1))

    @property
    def dask_workers_per_node(self):
        return self.data.get("processing", {}).get("dask_workers_per_node", 1)

    @property
    def dask_threads_per_worker(self):
        return self.data.get("processing", {}).get("dask_threads_per_worker", 1)

    @property
    def dask_memory_limit(self):
        return self.data.get("processing", {}).get("dask_memory_limit", "4GB")

    @property
    def peak_finder(self):
        peak_finder = self.data["piv"].get("peak_finder").lower()
        if peak_finder == "gauss3":
            peak_finder = 3
        elif peak_finder == "gauss4":
            peak_finder = 4
        elif peak_finder == "gauss5":
            peak_finder = 5
        elif peak_finder == "gauss6":
            peak_finder = 6
        else:
            raise ValueError(
                f"Invalid peak_finder: {peak_finder}. Must be 'gauss3', 'gauss4', or 'gauss5'."
            )
        return peak_finder

    @property
    def ensemble_piv(self):
        return self.ensemble

    @property
    def secondary_peak(self):
        return self.data["piv"]["secondary_peak"]

    @property
    def log_file(self) -> str:
        return self.data.get("logging", {}).get("file", "pypiv.log")

    @property
    def log_level(self) -> str:
        return self.data.get("logging", {}).get("level", "INFO").upper()

    @property
    def log_console(self) -> bool:
        return self.data.get("logging", {}).get("console", True)

    def _setup_logging(self):
        log_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(
            filename=self.log_file,
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        if self.log_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logging.getLogger().addHandler(console_handler)

        logging.info(
            "Logging initialized. Level: %s, File: %s", self.log_level, self.log_file
        )
