import numpy as np


def calc_gradient(d_j, d_k):
    """Calculate the gradient based on https://arxiv.org/pdf/1809.08993.pdf .
        Args: d_j is the first point , d_k is the following
        Returns: the relative steep between two points
    """
    delta_z_jk = d_k[2] - d_j[2]
    r_dj = np.hypot(d_j[0], d_j[1])
    r_dk = np.hypot(d_k[0], d_k[1])
    if r_dk - r_dj != 0:
        return np.arctan(delta_z_jk / (r_dk - r_dj))
    else:
        return 0

    # steep    shift
    #  10        0.8
    #  16        0.4
def calc_probability(grad, steep=16, shift=0.4):
    # @param
    """Apply a probability function based on the return -tanh- .
        Args: grad is the gradient value, a is steep, b is shift
        Returns: the probability of a given gradient value to be a border
    """
    return (1 + np.tanh(steep * (grad - shift))) / 2


def transform_spherical_to_cartesian(pts_sph):
    """Generates an array of spherical coordinates.
          Args:
            pts_sph: Array of cartesian coordinates in form of [N,(x,y,z)].
          Returns:
            An array in shape of pts_cart in spherical coordinates with inner dims: radius, azimuth, elevation in rad.
    """
    pts_cart = np.zeros(pts_sph.shape)
    for i in range(len(pts_cart)):
        r = pts_sph[i][0]
        az = pts_sph[i][1]
        el = pts_sph[i][2]
        pts_cart[i] = sph2cart(r, az, el)
    return pts_cart


def transform_cartesian_to_spherical(pts_cart):
    """Generates an array of spherical coordinates.
          Args:
            pts_cart: Array of cartesian coordinates in form of [N,(x,y,z)].
          Returns:
            An array in shape of pts_cart in spherical coordinates with inner dims: radius, azimuth, elevation in rad.
    """
    pts_sph = np.zeros(pts_cart.shape)
    for i in range(len(pts_cart)):
        x = pts_cart[i][0]
        y = pts_cart[i][1]
        z = pts_cart[i][2]
        pts_sph[i] = cart2sph(x, y, z)
    return pts_sph


def sph2cart(r, az, el):
    """calculation of spherical coordinates into cartesian coordinates .
        Returns: three values in form [x, y, z]
    """
    r_cos_theta = r * np.cos(el)
    x = r_cos_theta * np.cos(az)
    y = r_cos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def cart2sph(x, y, z):
    """calculation of cartesian coordinates into spherical coordinates .
        Returns: three values in form [r, az, el]
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el
