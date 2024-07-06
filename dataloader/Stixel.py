import yaml
from libraries.names import StixelClass
# 0.1 Load configfile
with open('config.yaml') as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


class BaseStixel:
    def __init__(self, column=None, top_row=None, bottom_row=None, depth=42.0, grid_step=config['grid_step']):
        self.column = column
        self.top_row = top_row
        self.bottom_row = bottom_row
        self.position_class: StixelClass = StixelClass.TOP
        self.depth = depth
        self.grid_step = grid_step
        self.scale_by_grid()

    def __repr__(self):
        return f"{self.column},{self.top_row},{self.bottom_row},{self.depth}"

    def scale_by_grid(self):
        self.column = int(self.column * self.grid_step)
        self.top_row = int(self.top_row * self.grid_step)
        self.bottom_row = int(self.bottom_row * self.grid_step)
