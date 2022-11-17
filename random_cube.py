from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np
from stl import mesh


# helper classes
class Orientation(Enum):
    DOWN = "down"
    UP = "up"
    LEFT = "left"
    RIGHT = "right"
    INTO = "into"
    OUT_OF = "out-of"

    def get_neighbour(self, grid, x, y, z):
        if self is Orientation.UP and y != len(grid[x])-1:
            return grid[x][y+1][z]
        if self is Orientation.DOWN and y != 0:
            return grid[x][y-1][z]
        if self is Orientation.LEFT and x != 0:
            return grid[x-1][y][z]
        if self is Orientation.RIGHT and x != len(grid)-1:
            return grid[x+1][y][z]
        if self is Orientation.INTO and z != len(grid[x][y])-1:
            return grid[x][y][z+1]
        if self is Orientation.OUT_OF and z != 0:
            return grid[x][y][z-1]
        return None

    def reverse(self):
        if self is Orientation.DOWN:
            return Orientation.UP
        if self is Orientation.UP:
            return Orientation.DOWN
        if self is Orientation.LEFT:
            return Orientation.RIGHT
        if self is Orientation.RIGHT:
            return Orientation.LEFT
        if self is Orientation.INTO:
            return Orientation.OUT_OF
        if self is Orientation.OUT_OF:
            return Orientation.INTO


@dataclass
class Rectangle:
    ll : np.ndarray
    tl : np.ndarray
    lr : np.ndarray
    tr : np.ndarray
    orientation : Orientation

    def faces(self):
        if self.orientation in [Orientation.UP, Orientation.RIGHT, Orientation.OUT_OF]:
            # right hand rule counterclockwise
            return [self.ll, self.tr, self.tl], [self.ll, self.lr, self.tr]
        # right hand rule clockwise
        return [self.ll, self.tl, self.tr], [self.ll, self.tr, self.lr]


@dataclass
class Prism:
    vx_000 : np.ndarray
    vx_001 : np.ndarray
    vx_010 : np.ndarray
    vx_011 : np.ndarray
    vx_100 : np.ndarray
    vx_101 : np.ndarray
    vx_110 : np.ndarray
    vx_111 : np.ndarray

    def triangle_slice(self, suppress_orientations=None):
        suppress_orientations = suppress_orientations or set()
        rectangles = [
            # left / right faces
            # x = 0
            Rectangle(self.vx_000, self.vx_001, self.vx_010, self.vx_011, Orientation.LEFT),
            # x = 1
            Rectangle(self.vx_100, self.vx_101, self.vx_110, self.vx_111, Orientation.RIGHT),
            # up / down faces
            # z = 1
            Rectangle(self.vx_001, self.vx_011, self.vx_101, self.vx_111, Orientation.UP),
            # z = 0
            Rectangle(self.vx_000, self.vx_010, self.vx_100, self.vx_110, Orientation.DOWN),
            # into / out of faces
            # y = 1
            Rectangle(self.vx_010, self.vx_011, self.vx_110, self.vx_111, Orientation.INTO),
            # y = 0
            Rectangle(self.vx_000, self.vx_001, self.vx_100, self.vx_101, Orientation.OUT_OF)
        ]
        return [
            f for r in rectangles
            if r.orientation not in suppress_orientations
            for f in r.faces()
        ]

    def volume(self):
        return (
            (self.vx_100[0] - self.vx_000[0]) *
            (self.vx_010[1] - self.vx_000[1]) *
            (self.vx_001[2] - self.vx_000[2])
        )


@dataclass
class PrismGraphNode:
    prism: Prism
    neighbour_map: Dict[Orientation, Prism]


# Inspired by quadtrees: https://en.wikipedia.org/wiki/Quadtree
class PrismGraph:
    def __init__(self, x_grid, y_grid, z_grid):
        prism_grid = [
            [
                [
                    Prism(
                        (x, y, z),
                        (x, y, z_next),
                        (x, y_next, z),
                        (x, y_next, z_next),
                        (x_next, y, z),
                        (x_next, y, z_next),
                        (x_next, y_next, z),
                        (x_next, y_next, z_next)
                    )
                    for z, z_next in zip(z_grid, z_grid[1:])
                ]
                for y, y_next in zip(y_grid, y_grid[1:])
            ]
            for x, x_next in zip(x_grid, x_grid[1:])
        ]
        self.graph = [
            [
                [
                    PrismGraphNode(
                        prism_grid[x][y][z],
                        {
                            o: o.get_neighbour(prism_grid, x, y, z)
                            for o in Orientation
                            if o.get_neighbour(prism_grid, x, y, z)
                        }
                    )
                    for z in range(len(prism_grid[x][y]))
                ]
                for y in range(len(prism_grid[x]))
            ]
            for x in range(len(prism_grid))
        ]

    def delete_prism(self, x, y, z):
        if self.graph[x][y][z].prism is None:
            return 0

        volume = self.graph[x][y][z].prism.volume()
        self.graph[x][y][z].prism = None
        for o in Orientation:
            neighbour_node = o.get_neighbour(self.graph, x, y, z)
            if neighbour_node:
                neighbour_node.neighbour_map.pop(o.reverse(), None)
        return volume

    def to_mesh(self):
        faces_vec = [
            face
            for n_x in self.graph
            for n_y in n_x
            for node in n_y
            if node.prism
            for face in node.prism.triangle_slice(
                suppress_orientations=node.neighbour_map.values())
        ]
        cube = mesh.Mesh(np.zeros(len(faces_vec), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces_vec):
            cube.vectors[i, :] = f
        return cube


if __name__ == "__main__":
    np.random.seed(0)
    N_DIVS = 27

    x_grid = np.linspace(0, 1, N_DIVS)
    y_grid = np.linspace(0, 1, N_DIVS)
    z_grid = np.linspace(0, 1, N_DIVS)

    desired_volume = (20 / 27) ** 3
    current_volume = 1.0

    g = PrismGraph(x_grid, y_grid, z_grid)

    # spray and pray?
    while current_volume > desired_volume:
        a = np.random.randint(1, len(g.graph)-1)
        b = np.random.randint(1, len(g.graph)-2)

        for x in range(len(g.graph)):
            current_volume -= g.delete_prism(x, a, b)
        for y in range(len(g.graph)):
            current_volume -= g.delete_prism(a, y, b)
        for z in range(len(g.graph)):
            current_volume -= g.delete_prism(a, b, z)

    cube = g.to_mesh()
    cube.save('cube_removed.stl')