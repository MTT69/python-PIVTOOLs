# Filters that require batches of images to operate correctly
BATCH_FILTERS = {"time", "pod"}


def requires_batch(filter_type: str) -> bool:
    """
    Check if a filter requires batches of images to operate.

    Args:
        filter_type (str): Type of filter (e.g., 'time', 'pod', 'gaussian')

    Returns:
        bool: True if filter needs multiple images, False otherwise
    """
    return filter_type in BATCH_FILTERS
