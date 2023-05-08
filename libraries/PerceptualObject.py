from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

grid = 8

def force_value_into_grid(value, grid_size=grid):
    rest = value % grid_size
    if rest >= grid / 2:
        value = value + (grid_size - rest)
    else:
        value = value - rest
    return value


def mean_double_row_values(depth_vals):
    """
    Searches for doubled rows and correct them to the mean of both
    Returns: a list with row and depth value without duplicates and cols [..., (row, depth)]
    """
    depth_vals_with_duplicates = []
    for pt in depth_vals:
        # row we are talking about
        row = pt[1]
        double_pt = []
        # search along all elements (incl. the row itself)
        for other_pt in depth_vals:
            # if the row is multiple times available (at least one time, itself)...
            if row == other_pt[1]:
                # ...add it
                double_pt.append(other_pt[2])
        # apply the mean of all found duplicates
        depth_vals_with_duplicates.append((row, mean(double_pt)))
    # add row and depth to a new list
    seen = set()
    uniq = []
    for x in depth_vals_with_duplicates:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


class PerceptualObject(object):
    def __init__(self, points):
        self.points = points



class ObjectColumn(object):
    def __init__(self, col=0, start=0, end=0, obj_pts=None):
        """
        Analyses one col of a given object cluster
        Args:
            col:
            start:
            end:
            obj_pts:
        """
        self.col = col
        self.starting_row = force_value_into_grid(start)   # from top to bottom
        self.ending_row = force_value_into_grid(end)       # lower than starting
        self. depth_values = []                            # (row, depth)
        depth_vals = obj_pts
        self.depth_values = mean_double_row_values(depth_vals)
        self.stixels = []                                  # (col, row, dist)

    def print_all(self):
        print("Col: " + str(self.col))
        print("Start: " + str(self.starting_row))
        print("End: " + str(self.ending_row))
        print("Pts: ")
        print(self.depth_values)

    def print_object_profile(self, marker_size=10):
        """
        Prints a 2D Graph of the object shape
        :param marker_size:
        :return:
        """
        if not self.depth_values:
            print("Object from " + str(self.ending_row) + " to " + str(self.starting_row) + " has no Depth Data.")
            return None
        xs = []
        ys = []
        x0 = []
        y0 = []
        for pt in self.depth_values:
            xs.append(pt[1])    # add depth to x-axle
            ys.append(pt[0])    # add row to y-axle
        # add detected Stixel if available
        if self.stixels:
            for stixel_pt in self.stixels:
                for i in range(len(ys)):
                    if stixel_pt[1] == ys[i]:
                        # print("Stixel found and added: " + str(stixel_pt[0]))
                        x0.append(xs[i])  # width, col
                        y0.append(stixel_pt[1])  # depth_val, cause of the presentation
                        break

        plt.figure(figsize=(15, 10))
        plt.plot(xs, ys, markersize=marker_size/10)
        plt.scatter(xs, ys, c='blue', s=marker_size)
        plt.scatter(x0, y0, c='red', s=marker_size * 3)

        # add a caption per point
        for x, y, g in zip(xs, ys, ys):
            if y in y0:
                label = "Stixel@{}|{}".format(int(g), round(x, 2))
            else:
                label = "{}|{}".format(int(g), round(x, 2))
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(12, 10),  # distance from text to points (x,y)
                         size=10,
                         ha='left')
        # change axis dims
        max_depth = max(self.depth_values, key=lambda x: x[1])
        min_depth = min(self.depth_values, key=lambda x: x[1])
        plt.axis([np.floor(min_depth[1]), np.ceil(max_depth[1]), self.starting_row+8, self.ending_row-8])

        plt.title("Col: " + str(self.col) + ", starting at: " + str(self.ending_row) + ", ending at:" + str(self.starting_row))
        plt.show()
        return None

    def apply_anomaly_detection_and_linear_completion(self, scam_threshold=1.1, grid_size=grid):
        """
        this function compares the object dimension on the image with the existing laser measurements and searches for
        missing data like holes, reflection, windows etc. it also applies a plausibility check and deletes scam pts.
        Args:
            scam_threshold: the value in m when the cut/jump ist too high to be a proper measurement of a
            contiguous object
            grid_size: defines the grid which needs to be interpolated
        Returns: an object-list of point-lists shape: [..., ([..., (x, y, d)])] with checked measurements and completed
        object laser data (linear)
        """
        if self.depth_values:
            fill_pts = []
            pt_list = np.asarray(self.depth_values)
            for pt in range(self.starting_row, self.ending_row-1, -grid_size):
                print(pt)
                if pt not in pt_list[:, 0]:
                    """ if pt before doesn't exist - e.g. 816= 824 (not existing) """
                    if pt+grid_size not in pt_list[:, 0]:
                        # search for the next fitting pt and overtake the vals
                        for ref_pt in range(pt, self.ending_row, -grid_size):
                            if ref_pt - grid_size in pt_list[:, 0]:
                                new_pt = (pt, pt_list[np.where(pt_list[:, 0] == (ref_pt - grid_size))[0][0]][1])
                                fill_pts.append(new_pt)
                                break
                    """ if the pt after doesn't exist - e.g. 680 = 680 (not existing but in object) """
                    if pt-grid_size not in pt_list[:, 0]:
                        # search for the next fitting pt and overtake the vals
                        for ref_pt in range(pt, self.starting_row, grid_size):
                            if ref_pt + grid_size in pt_list[:, 0]:
                                new_pt = (pt, pt_list[np.where(pt_list[:, 0] == (ref_pt + grid_size))[0][0]][1])
                                fill_pts.append(new_pt)
                                break
                    """ if both pts (next and before) exist: mean """
                    if pt + grid_size in pt_list[:, 0] and pt - grid_size in pt_list[:, 0]:
                        dist_1 = pt_list[np.where(pt_list[:, 0] == (pt + grid_size))[0][0]][1]
                        dist_2 = pt_list[np.where(pt_list[:, 0] == (pt - grid_size))[0][0]][1]
                        new_pt = (pt, np.mean((dist_1, dist_2)))
                        fill_pts.append(new_pt)
            for pt in fill_pts:
                self.depth_values.append(pt)
            # sort depth vals
            self.depth_values = sorted(self.depth_values, key=lambda x: x[0])
            self.check_vals_for_plausibility()
            return self.depth_values

    def check_vals_for_plausibility(self, scam_threshold=0.8):
        """
        This method checks a depth profile for unplausible values like windows in cars e.g.
        Args:
            scam_threshold: descibes the thresold when an unplausible value is detected
        Returns: a list of the adapted depth values (from the object)
        """
        self.depth_values = np.asarray(self.depth_values)
        for i in range(len(self.depth_values)-1):
            # e.g. 753.0 - 723.0 = 30.0 > 1.1: divide by two
            # e.g. 2: 723.0 - 753.0 = -30 > 1.1 : False
            diff = self.depth_values[i+1][1] - self.depth_values[i][1]
            while diff > scam_threshold:
                self.depth_values[i+1][1] = self.depth_values[i+1][1]-abs(diff/2)
                print(self.depth_values[i][1])
                print(self.depth_values[i+1][1])
                print("Px: " + str(self.depth_values[i+1][0]) + " will be adapted to " + str(self.depth_values[i+1][1]))
                diff = self.depth_values[i+1][1] - self.depth_values[i][1]
        self.depth_values = list(self.depth_values)
        return self.depth_values

    def get_stixel_from_depth_profile(self, new_stixel_threshold=0.325):
        """
        Analyses a depth profile to extract stixel from it and returns them as a list.
        Args:
            new_stixel_threshold: describes the minimum value to detect a new stixel (depth gap)
        Returns: Stixel in form of 2D-x|y coordinate pairs
        """
        for pt in self.depth_values:
            if self.stixels:
                if self.stixels[-1][2] - new_stixel_threshold > pt[1]:
                    self.stixels.append((self.col, pt[0], pt[1]))
            else:
                self.stixels.append((self.col, pt[0], pt[1]))
        return self.stixels

    def find_more_stixel_on_segmentation(self):
        pass

