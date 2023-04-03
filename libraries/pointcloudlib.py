import numpy as np


def get_stixel_from_laser_data(laser_points_by_view):
    """
    Calculates object cuts (stixel) based on laser points over multiple views
    Args:
        laser_points_by_view: An array of laser points by view with cartesian coordinates + their projection [..., [x,y,z, img_x, img_y]]
    Returns: An Array (views) of stixel found by the laser data
    """
    laser_stixel = []
    # calculate over all views
    for view_laser_points in laser_points_by_view:
        laser_points_by_angle = __order_points_by_angle(view_laser_points)
        gradient_cuts = []
        # calculate over all columns
        for angle_points in laser_points_by_angle:
            gradient_cuts.append(__extract_object_cuts_from_an_angle(angle_points))
        laser_stixel.append(gradient_cuts)
    return laser_stixel


def __order_points_by_angle(laser_points):
    """
    Takes a list of laser points, convert them to spherical, orders them and returns sorted cartesian coordinates
    Args:
        laser_points: A list of laser points with cartesian coordinates + their projection to the related image [..., [x,y,z, img_x, img_y]]
    Returns: a list where every entry is a list of points ([..., [x,y,z, img_x, img_y]]) of the same angle
    """
    return []

def


def __extract_object_cuts_from_an_angle(points_by_angle, research_active=False):
    """ This function takes a list of points to iterate over the angle/ column to find the highest steep between
        relative points to spot out a possible object border (the lower one).
                  Args:
                    angle: the angle from the angle_indices list in number format >just for information<
                    research_active: if True print probability list with gradient values, if False nothing
                  Returns:
                    An array with calculated (estimated) borders of objects
    """
    angle_indices = self.__get_index_list_by_angle(angle)
    pts_angle_list = []
    for index in angle_indices:
        pts_angle_list.append(self.points[index])
    pts_angle_list = np.column_stack((pts_angle_list, angle_indices))
    # calculate gradient
    gradient_list = []
    for i in range(len(pts_angle_list) - 1):
        gradient_list.append(np.absolute(funcLib3D.calc_gradient(pts_angle_list[i], pts_angle_list[i + 1])))
    gradient_list.append(0)
    pts_angle_list = np.column_stack((pts_angle_list, gradient_list))
    # calculate border probability
    prob = []
    for value in gradient_list:
        prob.append(funcLib3D.calc_probability(value))
    pts_angle_list = np.column_stack((pts_angle_list, prob))
    if research_active:
        for i in range(len(gradient_list)):
            print("{idx}__  \tg:{grad} \t\tP:{prob}".format(idx=i, grad=round(gradient_list[i], 5),
                                                            prob=round(prob[i], 5)))
    # extract high probabilities
    high_prob_points = []
    for i in range(len(pts_angle_list)):
        # @param
        if pts_angle_list[i][5] >= 0.96:
            high_prob_points.append(pts_angle_list[i])

    # cluster objects based on x-difference
    if high_prob_points:
        object_cluster_threshold = 4.0
        object_list = []
        high_prob_points = np.array(high_prob_points)
        object_list.append(high_prob_points[0])
        for pt in high_prob_points:
            if pt[0] - object_cluster_threshold >= object_list[-1][0]:
                object_list.append(pt)
        objects = np.array(object_list)
        if research_active:
            return objects, pts_angle_list
        else:
            return objects
    else:
        if research_active:
            print("no border found at: " + str(angle))
            return np.array([]), np.array([])
        return np.array([])


def __transform_spherical_to_cartesian(pts_sph):
    """
    Generates an array of spherical coordinates.
    Args:
        pts_sph: Array of cartesian coordinates in form of [N,(x,y,z)].
    Returns: An array in shape of pts_cart in spherical coordinates with inner dims: radius, azimuth, elevation in rad.
    """
    pts_cart = np.zeros(pts_sph.shape)
    for i in range(len(pts_cart)):
        r = pts_sph[i][0]
        az = pts_sph[i][1]
        el = pts_sph[i][2]
        pts_cart[i] = __sph2cart(r, az, el)
    return pts_cart


def __transform_cartesian_to_spherical(pts_cart):
    """
    Generates an array of spherical coordinates.
    Args:
        pts_cart: Array of cartesian coordinates in form of [N,(x,y,z)].
    Returns: An array in shape of pts_cart in spherical coordinates with inner dims: radius, azimuth, elevation in rad.
    """
    pts_sph = np.zeros(pts_cart.shape)
    for i in range(len(pts_cart)):
        x = pts_cart[i][0]
        y = pts_cart[i][1]
        z = pts_cart[i][2]
        pts_sph[i] = __cart2sph(x, y, z)
    return pts_sph


def __sph2cart(r, az, el):
    """calculation of spherical coordinates into cartesian coordinates .
        Returns: three values in form [x, y, z]
    """
    r_cos_theta = r * np.cos(el)
    x = r_cos_theta * np.cos(az)
    y = r_cos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def __cart2sph(x, y, z):
    """calculation of cartesian coordinates into spherical coordinates .
        Returns: three values in form [r, az, el]
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el
