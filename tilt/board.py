from dataclasses import dataclass, field 
from typing import NamedTuple, Self 
from queue import Queue
from enum import Enum 
from abc import ABC 

# TODO: Improve PEP8 compliance 

# TODO: Not using polyomino nor concrete mesh types at the simulation level. They are only useful
# in specific optimizations involving the rendering of the board and certain algorithms.
# Will prefer and prioritize the use of caching patterns to separate optimization caches from
# the fundamental data.

# All comments are self-notes or TODOs and most should be cleared or replaced with doc comments eventually 

class Vec2D(NamedTuple):
    """A 2D position."""
    x: int 
    y: int 

    def add(self, other: Self):
        """Return the sum of this vector and `other`."""
        return Vec2D(x=self.x + other.x, y=self.y + other.y)


UNIT_VEC_NORTH: Vec2D = Vec2D(x=0, y=-1) 
UNIT_VEC_EAST: Vec2D = Vec2D(x=1, y=0)
UNIT_VEC_SOUTH: Vec2D = Vec2D(x=0, y=1)
UNIT_VEC_WEST: Vec2D = Vec2D(x=-1, y=0)


class GlueType(Enum):
    """A glue type, or no glue at all."""
    NONE = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4





class GlueSides(NamedTuple):
    """A glue denotes a bond between two tiles. Tiles which are glue-bound never loose their relationship."""
    north: GlueType = GlueType.NONE 
    east: GlueType = GlueType.NONE 
    south: GlueType = GlueType.NONE 
    west: GlueType = GlueType.NONE 


@dataclass(frozen=False, slots=True) 
class Tile(ABC):
    """Base class for tiles."""
    pos: Vec2D 


def tile_uid_generator():
    uid = 1
    while True:
        yield uid 
        uid += 1

next_uid = tile_uid_generator() 

def get_next_uid() -> int:
    global next_uid 
    return next(next_uid)


@dataclass(frozen=False, slots=True) 
class Singleton(Tile):
    """
    A singleton is a tile whose position changes in tilt, and may attach itself
    to other singletons which share adjacent glues. 
    """
    uid: int = field(default_factory=get_next_uid)
    glues: GlueSides = field(default_factory=GlueSides)


@dataclass(frozen=False, slots=True)
class TileList:
    tiles: dict[Vec2D, Singleton] = field(default_factory=dict[Vec2D, Singleton])

    def set_location_of(self, tile: Singleton, to: Vec2D):
        if tile.pos not in self.tiles:
            ... # error?
        
        self.tiles.pop(tile.pos)
        tile.pos = to 
        self.tiles[tile.pos] = tile 

    def add(self, tile: Singleton):
        assert not tile.pos in self.tiles 

        self.tiles[tile.pos] = tile 


@dataclass 
class Board:
    """A Board is a 2D space-bounded map which comprises of singleton tiles."""
    # TODO: Use sparse set if dict iteration proves to be significantly slower than list iteration 
    # Side note: Dictionary iteration is not significantly slower in perftest.py, but that may
    # change for sufficiently complex keys  
    size: Vec2D 
    tile_list: TileList = field(default_factory=TileList)
    concretes: set[Vec2D] = field(default_factory=set[Vec2D])


class TiltDirection(Enum):
    NORTH = 1 
    EAST = 2 
    SOUTH = 3 
    WEST = 4 


# TODO: How to index polyominoes? <-- caching question

# There are some guarantees we can make about polyominoes: 
# 
# 1. When a polyomino forms by a bonding of two glues, this bond will never
#    break. Thus, a polyomino may never *SHRINK* in size, though it may grow.
# 
# 2. During steps/tumbling, the movement of a polyomino is terminated by the 
#    first tile which cannot move further. Thus, only checking "delta edge"
#    tiles is necessary for testing collision, e.g., if a 2-column poly is
#    moving to the right, only tiles in the right column need be checked.
#
# Concretes
# 1. Concrete tiles do not need representation in memory as tile objects. Since
#    doing so is a waste of memory and time, I can simply store the concretes
#    as a list of locations which contain a concrete, then query that list when
#    perform relocation availability checking during tilt. 
# 2. When colliding with a concrete, it is possible to group a collection of
#    unbinded polyominoes as a single bounding box, to further minimize
#    recurses and bounds checking per-tile. 

# Questions to ask once scalability becomes paramount:
# 1. How can additonal features be added without bloat? For instance, when
#    Factory Mode is introduced to this edition of TumbleTiles. 

class TumbleController:
    """A utility driver which drives simulation of full tilt on a specified board."""
    @dataclass
    class Polyomino:
        """A set of tiles joined by a glues."""
        # TODO: Store redundant references to tiles on each delta edge to 
        #       minimize iteration time of polyominoes each step 
        singletons: list[Singleton] = field(default_factory=list[Singleton])

    def __init__(self, board: Board):
        self.board = board 

        # TODO: How can we perform cache invalidation faster than linear time 
        # without clearing cache completely? 
        self.polyomino_cache: set[TumbleController.Polyomino] = set() # Lazily loaded 
        self.tiles_visited_cache: set[int] = set() # Reset every step 
        self.polyominoes_visited_cache: set[TumbleController.Polyomino] = set() 
        self.tile_reposition_queue: Queue[tuple[Singleton, Vec2D]] = Queue() # TODO: Replace with named tuple

    def get_polyomino(self, tile: Singleton) -> Polyomino:
        """Retrieve the effective polyomino generated by bonds between singleton tiles."""
        connected_tiles: dict[Vec2D, Singleton] = {}

        tiles_to_visit: Queue[Singleton] = Queue()
        tiles_to_visit.put(tile)

        while not tiles_to_visit.empty():
            tile = tiles_to_visit.get()
            if tile.pos in connected_tiles: continue 

            connected_tiles[tile.pos] = tile 

            [tiles_to_visit.put(neighbor) for neighbor in self.glued_neighbors(tile)]

        return TumbleController.Polyomino(list(connected_tiles.values()))

    # TODO: Should I compute the entire polyomino instead? Probably yes. Or just cache the polyominoes. 
    def glued_neighbors(self, tile: Singleton) -> list[Singleton]:
        """Retrieve all neighbors adjacent to tile which share glues."""

        result: list[Singleton] = [] 

        pos_north = tile.pos.add(UNIT_VEC_NORTH)
        pos_east = tile.pos.add(UNIT_VEC_EAST)
        pos_south = tile.pos.add(UNIT_VEC_SOUTH)
        pos_west = tile.pos.add(UNIT_VEC_WEST)

        neighbor_north = None if not pos_north in self.board.tile_list.tiles else self.board.tile_list.tiles[pos_north]
        neighbor_east = None if not pos_east in self.board.tile_list.tiles else self.board.tile_list.tiles[pos_east]
        neighbor_south = None if not pos_south in self.board.tile_list.tiles else self.board.tile_list.tiles[pos_south]
        neighbor_west = None if not pos_west in self.board.tile_list.tiles else self.board.tile_list.tiles[pos_west]

        if neighbor_north and neighbor_north.glues.south != GlueType.NONE and neighbor_north.glues.south == tile.glues.north:
            result.append(neighbor_north)
        
        if neighbor_east and neighbor_east.glues.west != GlueType.NONE and neighbor_east.glues.west == tile.glues.east:
            result.append(neighbor_east)

        if neighbor_south and neighbor_south.glues.north != GlueType.NONE and neighbor_south.glues.north == tile.glues.south:
            result.append(neighbor_south)

        if neighbor_west and neighbor_west.glues.east != GlueType.NONE and neighbor_west.glues.east == tile.glues.west:
            result.append(neighbor_west)

        return result 

    def mark_polyomino_elements_visited(self, polyomino: Polyomino):
        """Marks all tiles in this `polyomino` as visited."""
        for tile in polyomino.singletons:
            self.tiles_visited_cache.add(tile.uid)

    def position_is_available(self, pos: Vec2D) -> bool:
        """Returns true if `pos` is within the bounding box of `self.board` and does not contain a concrete."""
        return not pos in self.board.concretes \
            and pos.x > 0 and pos.y > 0 \
            and pos.x < self.board.size.x and pos.y < self.board.size.y 

    def step_tiles_recursively(self, tile: Singleton, delta: Vec2D):
        """Step given tile and any glued or collided tiles."""
        
        # base case: tile has already moved 
        if tile.uid in self.tiles_visited_cache:
            return 

        # 1. Get polyomino bounding box 
        polyomino = self.get_polyomino(tile)
        self.mark_polyomino_elements_visited(polyomino)

        for tile in polyomino.singletons:
            projected_next_tile_pos = tile.pos.add(delta)

            # 2. If any part of polyomino is projected to overlap a concrete or surpass a board edge, terminate.
            if not self.position_is_available(projected_next_tile_pos):
                return 

            # 3. For all edges of the polyomino that touch a singleton at the edge -delta, attempt to move tile+delta first
            if projected_next_tile_pos in self.board.tile_list.tiles:
                self.step_tiles_recursively(self.board.tile_list.tiles[projected_next_tile_pos], delta)

                if projected_next_tile_pos in self.board.tile_list.tiles:
                    return 

        # 4. Move all tiles in polyomino towards delta
        for tile in polyomino.singletons:
            next_pos = tile.pos.add(delta)
            self.tile_reposition_queue.put((tile, next_pos))

    def enqueue_tile(self, tile: Singleton, next_pos: Vec2D):
        self.tile_reposition_queue.put((tile, next_pos))

    def deque_tile_reposition(self):
        while not self.tile_reposition_queue.empty():
            tile, next_pos = self.tile_reposition_queue.get()
            
            if next_pos in self.board.tile_list.tiles: 
                self.board.tile_list.tiles.pop(next_pos) 
            
            tile.pos = next_pos 
            self.board.tile_list.tiles[tile.pos] = tile 

    def step(self, direction: TiltDirection):
        """Perform a single step of a full tilt sequence in a given direction."""

        match direction:
            case TiltDirection.NORTH:
                delta = UNIT_VEC_NORTH 
            case TiltDirection.EAST:
                delta = UNIT_VEC_EAST 
            case TiltDirection.SOUTH:
                delta = UNIT_VEC_SOUTH 
            case TiltDirection.WEST:
                delta = UNIT_VEC_WEST 

        for tile in self.board.tile_list.tiles.values():
            self.step_tiles_recursively(tile, delta)

        self.deque_tile_reposition()

    def tumble(self, direction: TiltDirection):
        """Perform a complete full tilt in a given direction."""

        ...


class BoardWriter:
    def __init__(self, board: Board): 
        self.board = board 
        ...

    def print(self): 
        print("- " * (self.board.size.x + 2))

        for y in range(self.board.size.x):
            print("|", end=' ')

            for x in range(self.board.size.y):
                if Vec2D(x, y) in self.board.tile_list.tiles:
                    print("O", end=' ')
                elif Vec2D(x, y) in self.board.concretes:
                    print("#", end=' ')
                else:
                    print(".", end=' ')

            print("|")

        print("- " * (self.board.size.x + 2))


if __name__ == "__main__":
    board = Board(size=Vec2D(15, 15))
    writer = BoardWriter(board)

    board.tile_list.add(Singleton(Vec2D(0, 0)))
    board.tile_list.add(Singleton(Vec2D(0, 1)))
    board.tile_list.add(Singleton(Vec2D(1, 1)))

    board.concretes.add(Vec2D(2, 0))
    board.concretes.add(Vec2D(4, 1))

    controller = TumbleController(board)
    controller.step(TiltDirection.EAST)

    # writer.print()