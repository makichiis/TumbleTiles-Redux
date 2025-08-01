from dataclasses import dataclass, field 
from typing import NamedTuple, Self 
from queue import Queue 
from enum import Enum 

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


def tile_uid_generator():
    uid = 1
    while True:
        yield uid 
        uid += 1

next_uid = tile_uid_generator() 

def get_next_uid() -> int:
    global next_uid 
    return next(next_uid)


# @dataclass(frozen=False, slots=True) 
# class Tile:
#     """
#     A tile is a tile whose position changes in tilt, and may attach itself
#     to other tiles which share adjacent glues. 
#     """
#     uid: int = field(default_factory=get_next_uid)
#     glues: GlueSides = field(default_factory=GlueSides)


@dataclass(slots=True, frozen=False)
class Tile:
    position: Vec2D
    glues: GlueSides = field(default_factory=GlueSides)
    _uid: int = field(default_factory=get_next_uid)

    @property 
    def uid(self) -> int: return self._uid 


@dataclass(slots=True)
class TileMap: # NOTE: How to ensure key and tile position are synchronized? 
    """A collection of tiles indexed by position."""
    elements: dict[Vec2D, Tile] = field(default_factory=dict[Vec2D, Tile]) 

    def add(self, tile: Tile):
        assert tile.position not in self.elements 
        self.elements[tile.position] = tile 

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
    """A Board is a 2D space-bounded map which comprises of tile tiles."""
    # TODO: Use sparse set if dict iteration proves to be significantly slower than list iteration 
    # Side note: Dictionary iteration is not significantly slower in perftest.py, but that may
    # change for sufficiently complex keys  
    size: Vec2D 
    tiles: TileMap = field(default_factory=TileMap)
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
        """A set of tiles joined by a glues. Local cache of managed board tiles."""
        # TODO: Store redundant references to tiles on each delta edge to 
        #       minimize iteration time of polyominoes each step 
        # tiles: dict[Vec2D, Tile] = field(default_factory=dict[Vec2D, Tile])
        tiles: TileMap = field(default_factory=TileMap)

    def __init__(self, board: Board):
        self._board: Board = board 

        self.polyomino_cache: dict[Vec2D, TumbleController.Polyomino] = {} 
        self.last_direction: TiltDirection | None = None 
        self.effective_wall_cache: set[Vec2D] = set() 

    # TODO: How can we perform cache invalidation faster than linear time 
    # without clearing cache completely? 

    @property 
    def board(self) -> Board: return self._board 

    def get_polyomino(self, tile: Tile) -> Polyomino:
        """Retrieve the effective polyomino generated by bonds between tile tiles."""
        connected_tiles: dict[Vec2D, Tile] = {}

        tiles_to_visit: Queue[Tile] = Queue()
        tiles_to_visit.put(tile)

        while not tiles_to_visit.empty():
            tile = tiles_to_visit.get()
            if tile.position in connected_tiles: continue 

            connected_tiles[tile.position] = tile 

            [tiles_to_visit.put(neighbor) for neighbor in self.glued_neighbors(tile)]

        return TumbleController.Polyomino(TileMap(connected_tiles))

    # TODO: Should I compute the entire polyomino instead? Probably yes. Or just cache the polyominoes. 
    def glued_neighbors(self, tile: Tile) -> list[Tile]:
        """Retrieve all neighbors adjacent to tile which share glues."""

        if tile.glues == GlueSides(GlueType.NONE, GlueType.NONE, GlueType.NONE, GlueType.NONE):
            return [] 

        result: list[Tile] = [] 

        pos_north = tile.position.add(UNIT_VEC_NORTH)
        pos_east = tile.position.add(UNIT_VEC_EAST)
        pos_south = tile.position.add(UNIT_VEC_SOUTH)
        pos_west = tile.position.add(UNIT_VEC_WEST)

        neighbor_north = None if not pos_north in self._board.tiles.elements else self._board.tiles.elements[pos_north]
        neighbor_east = None if not pos_east in self._board.tiles.elements else self._board.tiles.elements[pos_east]
        neighbor_south = None if not pos_south in self._board.tiles.elements else self._board.tiles.elements[pos_south]
        neighbor_west = None if not pos_west in self._board.tiles.elements else self._board.tiles.elements[pos_west]

        if neighbor_north and neighbor_north.glues.south != GlueType.NONE and neighbor_north.glues.south == tile.glues.north:
            result.append(neighbor_north)
        
        if neighbor_east and neighbor_east.glues.west != GlueType.NONE and neighbor_east.glues.west == tile.glues.east:
            result.append(neighbor_east)

        if neighbor_south and neighbor_south.glues.north != GlueType.NONE and neighbor_south.glues.north == tile.glues.south:
            result.append(neighbor_south)

        if neighbor_west and neighbor_west.glues.east != GlueType.NONE and neighbor_west.glues.east == tile.glues.west:
            result.append(neighbor_west)

        return result 

    def position_is_available(self, at: Vec2D) -> bool:
        """Returns true if `pos` is within the bounding box of `self.board` and does not contain a concrete."""
        return (not (at in self._board.concretes or at in self.effective_wall_cache)) \
            and at.x >= 0 and at.y >= 0 \
            and at.x < self._board.size.x and at.y < self._board.size.y 

    def get_delta(self, of: TiltDirection) -> Vec2D:
        match of:
            case TiltDirection.NORTH:
                return UNIT_VEC_NORTH 
            case TiltDirection.EAST:
                return UNIT_VEC_EAST 
            case TiltDirection.SOUTH:
                return UNIT_VEC_SOUTH 
            case TiltDirection.WEST:
                return UNIT_VEC_WEST 

    def _get_adjacent_polyominoes(self, of: Polyomino, at_face: Vec2D) -> list[Polyomino]:
        """Retrieve all polyominoes that touch `of` at each `at_face`."""
        
        result: list[TumbleController.Polyomino] = [] 

        for pos in of.tiles.elements.keys():
            polyomino = self.polyomino_cache.get(pos.add(at_face))
            if polyomino: result.append(polyomino)

        return result 

    def _refresh_wall_cache(self, for_direction: TiltDirection):
        """Refreshes the cache of locations treated as walls when polyominoes cannot move."""
        self.effective_wall_cache.clear()
        
        delta = self.get_delta(for_direction)
        polar_delta = Vec2D(-delta.x, -delta.y)

        queue: Queue[TumbleController.Polyomino] = Queue() # wall location queue

        # enqueue polyominoes adjacent to concretes and delta edge 
        for pos, polyomino in self.polyomino_cache.items():
            adjacent_pos = pos.add(delta)
            if not self.position_is_available(adjacent_pos):
                queue.put(polyomino)

        # deque wall locations and cache, enqueue
        while not queue.empty():
            polyomino = queue.get()

            for tile in polyomino.tiles:
                self.effective_wall_cache.add(tile.position)

            for adjacent_polyomino in self._get_adjacent_polyominoes(polyomino, polar_delta):
                if not next(iter(adjacent_polyomino.tiles.elements.keys())) in self.effective_wall_cache:
                    queue.put(adjacent_polyomino) 

    def _refresh_polyomino_cache(self):
        self.polyomino_cache.clear() 
    
        for tile in self._board.tiles.elements.values():
            polyomino = self.get_polyomino(tile)

            for tile in polyomino.tiles.elements.values():
                self.polyomino_cache[tile.position] = polyomino

    def _update_board(self):
        """
        Refresh the contents of the managed board to reflect changes made within this controller.
        """
        self._board.tiles.elements.clear() 
        for polyomino in self.polyomino_cache.values():
            self._board.tiles.elements.update(polyomino.tiles.elements)
            polyomino.tiles.elements.clear() 

    def _step_impl(self, direction: TiltDirection): 
        """Internal logic."""
        # # if direction != self.last_direction: self.refresh_wall_cache(direction)
        # # else: self.update_wall_cache(direction)
        # self._refresh_polyomino_cache()
        self._refresh_wall_cache(direction)

        delta = self.get_delta(direction)

        for pos_initial, polyomino in self.polyomino_cache.items():
            mapped_position = next(iter(polyomino.tiles.elements.keys()))
            
            if mapped_position != pos_initial: continue 
            if mapped_position in self.effective_wall_cache: continue 

            for tile in polyomino.tiles.mut_iter():
                tile.position = tile.position.add(delta) 

        cache: dict[Vec2D, TumbleController.Polyomino] = {} 
        for polyomino in self.polyomino_cache.values():
            if next(iter(polyomino.tiles.elements.keys())) in cache: continue 

            for tile in polyomino.tiles:
                cache[tile.position] = polyomino 

        self.polyomino_cache = cache 

        # TODO: Polyomino cache invalidation (e.g., factory mode [?]) and merge glues 

    def step(self, direction: TiltDirection):
        """Perform a single step of a tilt in a given direction."""
        self._refresh_polyomino_cache()
        self._step_impl(direction)
        self._update_board()

    def tumble(self, direction: TiltDirection):
        """Perform a complete full tilt in a given direction."""
        self._refresh_polyomino_cache()

        if direction == TiltDirection.NORTH or direction == TiltDirection.SOUTH:
            for _ in range(self._board.size.y): 
                cache_len_old = len(self.effective_wall_cache)
                self._step_impl(direction) 
                if len(self.effective_wall_cache) == cache_len_old and cache_len_old != 0: break 
        else:
            for _ in range(self._board.size.x): 
                cache_len_old = len(self.effective_wall_cache)
                self._step_impl(direction) 
                if len(self.effective_wall_cache) == cache_len_old and cache_len_old != 0: break 
        
        self._update_board() 

    def run_sequence(self, of: list[TiltDirection]):
        """Perform a sequence of full tilts with directiosn given by a list of movements."""
        for direction in of: self.tumble(direction)

    def cycle(self, in_directions: list[TiltDirection], n_times: int):
        """Repeat a sequence of full tilts 0 or more times."""
        for _ in range(n_times): self.run_sequence(in_directions)


class BoardWriter:
    def __init__(self, board: Board): 
        self.board = board 

    def print(self): 
        print("- " * (self.board.size.x + 2))

        for y in range(self.board.size.y):
            print("|", end=' ')

            for x in range(self.board.size.x):
                if Vec2D(x, y) in self.board.tiles.elements:
                    print("O", end=' ')
                elif Vec2D(x, y) in self.board.concretes:
                    print("#", end=' ')
                else:
                    print(".", end=' ')

            print("|")

        print("- " * (self.board.size.x + 2))


if __name__ == "__main__":
    board = Board(size=Vec2D(8, 4))
    writer = BoardWriter(board)

    board.tiles.add(Tile(Vec2D(0, 0)))
    board.tiles.add(Tile(Vec2D(1, 0)))
    board.tiles.add(Tile(Vec2D(0, 1)))
    board.tiles.add(Tile(Vec2D(3, 1)))

    board.concretes.add(Vec2D(4, 0))
    board.concretes.add(Vec2D(6, 1))

    controller = TumbleController(board)
    print("Initial:")
    writer.print()

    controller.tumble(TiltDirection.EAST)
    writer.print()  

    controller.tumble(TiltDirection.SOUTH)
    writer.print() 

    controller.tumble(TiltDirection.WEST)
    writer.print() 

    controller.tumble(TiltDirection.NORTH)
    writer.print()
