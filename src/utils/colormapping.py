import numpy as np


def altitude_to_rgb(altitude, min_alt, max_alt):
    """
    Map a single altitude value to an RGB color using linear interpolation.

    :param altitude: float, the altitude value
    :param min_alt: minimum altitude for normalization
    :param max_alt: maximum altitude for normalization
    :return: (R, G, B) tuple with values 0-255
    """
    # Normalize to [0,1]
    t = np.clip((altitude - min_alt) / (max_alt - min_alt), 0, 1)

    stops = [
        (1.0, np.array([255, 255, 255])),  # white (highest)
        (0.6, np.array([139, 69, 19])),  # brown
        (0.2, np.array([0, 255, 0])),  # green
        (0.0, np.array([0, 100, 255])),  # blue (lowest)
    ]

    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t1 <= t <= t0:
            ratio = (t - t0) / (t1 - t0)
            r, g, b = (c0 + (c1 - c0) * ratio).astype(np.uint8)
            return int(r), int(g), int(b)

    return tuple(stops[-1][1])


def altitudes_to_rgb_array(altitudes, min_alt=None, max_alt=None):
    """
    Map a 2D altitude array to a 3D RGB array using linear interpolation.

    :param altitudes: 2D numpy array of altitudes
    :param min_alt: minimum altitude for normalization (optional)
    :param max_alt: maximum altitude for normalization (optional)
    :return: 3D numpy array (height, width, 3) with RGB values in 0-255
    """
    if min_alt is None:
        min_alt = altitudes.min()
    if max_alt is None:
        max_alt = altitudes.max()

    # Normalize altitudes to [0,1]
    t = np.clip((altitudes - min_alt) / (max_alt - min_alt), 0, 1)
    rgb = np.zeros((*altitudes.shape, 3), dtype=np.uint8)

    stops = [
        (1.0, np.array([255, 255, 255])),  # white (highest)
        (0.6, np.array([139, 69, 19])),  # brown
        (0.3, np.array([0, 255, 0])),  # green
        (0.0, np.array([0, 0, 255])),  # blue (lowest)
    ]

    # Interpolate between stops
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        mask = (t > t1) & (t < t0)
        ratio = (t[mask] - t0) / (t1 - t0)
        rgb[mask] = (c0 + (c1 - c0) * ratio[:, None]).astype(np.uint8)

    return rgb
