import numpy
import matplotlib.pyplot as plt

class Node:

    def __init__(self, points, parent, level):
        self.points = points
        self.parent = parent
        self.level = level
        self.left = None
        self.right = None
        self.split_orientation = ""
        self.split_coordinate = None

    def vertical_split(self):
        self.points.sort( key=lambda x: x[0] )
        mid_point = len(self.points) // 2
        
        self.split_orientation = "vertical"
        self.split_coordinate = self.points[mid_point]

        return (self.points[:mid_point], self.points[mid_point:])

    def horizontal_split(self):
        self.points.sort( key=lambda x: x[1] )
        mid_point = len(self.points) // 2

        self.split_orientation = "horizontal"
        self.split_coordinate = self.points[mid_point]

        return (self.points[:mid_point], self.points[mid_point:])

    def num_points(self):
        return len(self.points)


    def visualize_node(self, fig):
        if self.split_orientation == "vertical":
            fig.plot([self.split_coordinate[0], self.split_coordinate[1]-100], [self.split_coordinate[0], self.split_coordinate[1]+100], 'k-', lw=10-self.level)
        elif self.split_orientation == "horizontal":
            fig.plot([self.split_coordinate[0]-100, self.split_coordinate[0]], [self.split_coordinate[0]+100, self.split_coordinate[1]], 'k-', lw=10-self.level)




class KDTree:

    def __init__(self, points, max_depth = 10):
        self.points = points
        self.max_depth = max_depth
        self.root = Node(points, parent=None, level=0)
        self.levels = []



    def construct_tree(self):
        split_index = 0
        depth = 0
        self.levels.append( [self.root] )
        while depth < self.max_depth:
            level_paritions = self.levels[depth]

            newlevel_partitions = []
            for partition in level_paritions:
                if partition.num_points() > 1:
                    if depth % 2 == 0:
                        first_half, second_half = partition.vertical_split()
                    else:
                        first_half, second_half = partition.horizontal_split()

                    left_node = Node(first_half, partition, level=depth+1)
                    right_node = Node(second_half, partition, level=depth+1)

                    newlevel_partitions.append(left_node)
                    newlevel_partitions.append(right_node)

            if len(newlevel_partitions) == 0:
                break
            self.levels.append( newlevel_partitions )
            depth += 1


    # def visualize_tree(self, fig):
        





if __name__ == '__main__':
    test_points = [ (2, 4), (-2, 4), (2, -4), (-2, -4), (3, 8), (-6, 9) ]
    tree = KDTree(test_points)
    tree.construct_tree()






