class GlucoCareError(Exception):
    """Base class for domain errors that map to a 4xx response."""


class InsufficientDataError(GlucoCareError):
    """Not enough valid glucose data to compute the requested result."""
