from board import Board, Vec2D, Tile, TumbleController, BoardWriter, TiltDirection

from random import randint 

def generate_board(size: Vec2D) -> Board:
    """Create a board of a given `size` with tiles and concretes."""
    result: Board = Board(size)

    for x in range(size.x): 
        for y in range(size.y):
            roll = randint(1, 10)
            if roll > 7:
                result.tiles.add(Tile(Vec2D(x, y)))
            elif roll > 6:
                result.concretes.add(Vec2D(x, y))
            
    return result 

def print_board(board: Board):
    BoardWriter(board).print() 

north = TiltDirection.NORTH 
east = TiltDirection.EAST 
south = TiltDirection.SOUTH 
west = TiltDirection.WEST 

controller = TumbleController(Board(size=Vec2D(0, 0))) # shuts up linter 
