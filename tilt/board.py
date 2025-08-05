from dataclasses import dataclass, field 
from typing import NamedTuple, Self
from queue import Queue 
from enum import Enum


class Direction(Enum):
    """A cardinal direction in 2D. 
    
    Represents a cardinal direction, intentionally in tilt.
    The specific values of each enumerical value are
    implementation-defined and irrelevant to any end user.
    """
    north = 1
    east = 2
    south = 3 
    west = 4 


class Vec2D(NamedTuple):
    """A 2D vector. 
    
    A `Vec2D` represents a 2D position on a Cartesian plane. The `Vec2D` structure is
    used in many parts of this library. Vectors are immutable, thus, a new vector must be
    constructed to represent an existing vector with a modified component. 
    
    Attributes
    ----------
    x : `int` 
        The *x* component of a `(x, y)` vector. 

    y : `int` 
        The *y* component of a `(x, y)` vector. 
    """
    x: int 
    y: int 

    def add(self, other: Self):
        """Retrieve the sum of this vector and `other`.
        
        Parameters
        ----------
        other : `Vec2D` 
            The vector to add this vector to.

        Return
        ------
            A new `Vec2D` containing the sum of this vector and *`other`*. 
        """
        return Vec2D(x=self.x + other.x, y=self.y + other.y)


UNIT_VEC_NORTH: Vec2D = Vec2D(x=0, y=-1) 
UNIT_VEC_EAST: Vec2D = Vec2D(x=1, y=0)
UNIT_VEC_SOUTH: Vec2D = Vec2D(x=0, y=1)
UNIT_VEC_WEST: Vec2D = Vec2D(x=-1, y=0)


def unit_vec(direction: Direction) -> Vec2D:
    match direction:
        case Direction.north:
            return UNIT_VEC_NORTH
        case Direction.east:
            return UNIT_VEC_EAST
        case Direction.south:
            return UNIT_VEC_SOUTH
        case Direction.west:
            return UNIT_VEC_WEST

class Glue:
    """A glue type, or no glue at all."""
    none =  0
    north = 0b00000001
    east =  0b00000010
    south = 0b00000100
    west =  0b00001000


GlueMask = int 
class GlueMasks:
    north: int = 0xFF << 24
    east: int = 0xFF << 16
    south: int = 0xFF << 8
    west: int = 0xFF


def get_gluemask(north: int, east: int, south: int, west: int) -> GlueMask:
    return north << 24 | east << 16 | south << 8 | west 


# TODO: Pre-load initial UID on project import to prevent overlap (tile factory..?)
def tile_uid_generator(initial: int=1): 
    uid = initial 
    while True: 
        yield uid 
        uid += 1 

next_uid = tile_uid_generator() 

def get_next_uid() -> int:
    global next_uid 
    return next(next_uid)


@dataclass(slots=True, frozen=False)
class Tile:
    """An atomic mutable tile structure. 

    A tile is a pixel which may "move" across a board during reconfiguration 
    sequences. Any pixel on a board denoted by a tile is equally affected by
    global control, and may contain rules which govern whether it sticks to
    other pixels during simulation (denoted by `glues`).

    Attributes
    ----------
    position : `Vec2D` 
        The translation of this tile in any given space.
    
    glues : `GlueMask` 
        A glue mask represents the stickiness attribute of this tile. 
    
    uid : `int`
        The `uid` attribute of a tile is unique and persistent for the lifetime
        of the tile. The UID of a tile should never be end user-defined as it
        is a runtime utility attribute which allows a developer to distinguish 
        between tiles for any given purpose. 
        
        > The use of a UID vs. object IDs is so that metadata connected to this 
        tile remains persistent between sessions of a program using this tile 
        without the need for a user-implemented tile factory. 
    """

    position: Vec2D
    glues: GlueMask = Glue.none 
    _uid: int = field(default_factory=get_next_uid) # TODO: Make overridable/contexualized 

    @property 
    def uid(self) -> int: return self._uid 


@dataclass(slots=True)
class TileMap: # NOTE: How to ensure key and tile position are synchronized? 
    """A collection of tiles indexed by position."""
    _elements: dict[Vec2D, Tile] = field(default_factory=dict[Vec2D, Tile]) 

    @property 
    def elements(self) -> dict[Vec2D, Tile]: return self._elements 

    def add(self, tile: Tile):
        """
        Add a tile to this tile map. The tile map must not contain a tile
        which already exists at the position provided by `tile`. 
        """
        assert tile.position not in self.elements 
        self.elements[tile.position] = tile 

    def __len__(self) -> int: return len(self.elements)

    def __iter__(self): # TODO: Make custom immutable iterator ? 
        """
        Iterate over the tile selection without mutating position. Mutation of the position
        attribute will result in undefined behavior as index cache is not invalidated. 
        """
        return self.elements.values().__iter__()

    def mut_iter(self):
        """
        Iterate over the tile selection with the expectation that tiles will permutate.
        Behavior when this generator is terminated prematurely is undefined as it 
        invalidates and updates the index cache. 
        """
        changed_tiles: Queue[tuple[Tile, Vec2D]] = Queue()

        for position, tile in self.elements.items():
            yield tile 
            if position != tile.position: changed_tiles.put((tile, position))

        while not changed_tiles.empty():
            tile, old_position = changed_tiles.get()

            self.elements.pop(old_position)
            self.elements[tile.position] = tile  

    def set_position(self, of: Tile, to: Vec2D):
        assert self.elements[of.position] == of 
        
        self.elements.pop(of.position)
        of.position = to 
        self.elements[to] = of 


@dataclass 
class Board:
    """A 2D space-bounded map which comprises of tile tiles. 

    Attributes
    ----------
    size : `Vec2D`
        The effective size of this board. Data layout does not depend on this attribute,
        however may be utilized by a controller in a way that affects simulation output. 

    tiles : `TileMap`
        The collection of mutable tiles in this board. 

    concretes : `set[Vec2D]`
        The collection of concrete (wall) tiles in this board. 
    """
    # TODO: Use sparse set if dict iteration proves to be significantly slower than list iteration 
    # Side note: Dictionary iteration is not significantly slower in perftest.py, but that may
    # change for sufficiently complex keys  
    # TODO: Tile/Concrete occlusion checks 
    size: Vec2D 
    tiles: TileMap = field(default_factory=TileMap)
    concretes: set[Vec2D] = field(default_factory=set[Vec2D])

