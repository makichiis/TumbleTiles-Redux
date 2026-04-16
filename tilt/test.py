from board import *
from controller import * 

from random import randint 
from typing import Callable, Any 
import numpy as np 
import cv2 
from ffpyplayer.player import MediaPlayer # type: ignore 
from time import time
from typing import Union
import sys


def print_there(x: int, y: int, text: str):
    """Overwrites terminal buffer with `text` at `x, y`"""
    sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
    sys.stdout.flush()


class BoardWriter:
    """
    A utility class for rendering a reasonably-sized board to the stdout.

    This class serves as a writer utility for unit testing and debugging
    any given board simulator. 
    """
    def __init__(self, board: Board, use_ascii: bool=True): 
        self.board = board 
        self.use_ascii = use_ascii 

    def _print_impl_ascii(self):
        print("- " * (self.board.size.x + 2))

        for y in range(self.board.size.y):
            print("|", end=' ')

            for x in range(self.board.size.x): 
                pos = Vec2D(x, y) 
                if pos in self.board.tiles.elements:
                    print("O", end=' ')
                elif pos in self.board.concretes:
                    print("#", end=' ')
                else:
                    print(".", end=' ')

            print("|")

        print("- " * (self.board.size.x + 2))

    def _println(self, text: str, overdraw: bool=False):
        if overdraw:
            print_there(0, 1, text)
            print('')
        else:
            print(text)

    def _print_impl_utf8(self, overdraw: bool=False):
        # framebuffer: list[list[str]] = [[' '] * (self.board.size.x+2)] * (self.board.size.y+2) # mein gott 
        # possible_characters = [  
        #                         '', '╠', '╣', '╝', '╚', '╩', # north 
        #                         '╚', '╔', '╩', '╦', '╠', '═', # east 
        #                         '║', '╠', '╣', '╗', '╔', '╦', # south 
        #                         '╝', '╗', '╩', '╦', '╠', '═', # west 
        #                       ]
        
        buffer: str = ''

        buffer += '╔═'
        for x in range(self.board.size.x):
            if self.board.concretes.__contains__(Vec2D(x, 0)):
                buffer += '╦═'
            else:
                buffer += '══'
        buffer += '╗\n'

        for y in range(self.board.size.y):
            if Vec2D(0, y) in self.board.concretes:
                buffer += '╠═'
            else:
                buffer += '║ '

            for x in range(self.board.size.x):
                pos = Vec2D(x, y)

                if pos in self.board.concretes:
                    # NOTE: this is gonna be rough 
                    north = self.board.concretes.__contains__(pos.add(UNIT_VEC_NORTH))\
                            or pos.y == 0 
                    east = self.board.concretes.__contains__(pos.add(UNIT_VEC_EAST))\
                            or pos.x == self.board.size.x-1
                    south = self.board.concretes.__contains__(pos.add(UNIT_VEC_SOUTH))\
                            or pos.y == self.board.size.y-1
                    west = self.board.concretes.__contains__(pos.add(UNIT_VEC_WEST))\
                            or pos.x == 0 

                    if north and east and south and west:
                        buffer += '╬'+'═'
                    elif north and east and west:
                        buffer += '╩'+'═'
                    elif north and south and east:
                        buffer += '╠'+'═'
                    elif north and south and west:
                        buffer += '╣'+' '
                    elif south and east and west:
                        buffer += '╦'+'═'
                    elif north and east:
                        buffer += '╚'+'═'
                    elif north and west:
                        buffer += '╝'+' '
                    elif south and east:
                        buffer += '╔'+'═'
                    elif south and west:
                        buffer += '╗'+' '
                    elif north and south:
                        buffer += '║'+' '
                    elif north:
                        buffer += '╨'+' '
                    elif south:
                        buffer += '╥'+' '
                    elif east and west:
                        buffer += '═'+'═'
                    elif east:
                        buffer += '╞'+'═'
                        ...
                    elif west:
                        buffer += '╡'+' '
                        ...
                    else:
                        buffer += '■'+' '
                elif pos in self.board.tiles.elements:
                    buffer += '□'+' '
                else:
                    buffer += '⋅'+' '

            if Vec2D(self.board.size.x-1, y) in self.board.concretes:
                buffer += '╣\n'
            else:
                buffer += '║\n'

        buffer += '╚'+'═'
        for x in range(self.board.size.x):
            if self.board.concretes.__contains__(Vec2D(x, self.board.size.y-1)):
                buffer += '╩'+'═'
            else:
                buffer += '═'+'═'
        buffer += '╝\n'

        self._println(buffer, overdraw)

    def print(self, overdraw: bool=False): 
        """Write the contents of the given `board` to stdout."""

        if self.use_ascii: self._print_impl_ascii()
        else: self._print_impl_utf8(overdraw)

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

def print_board(board: Board, use_ascii: bool=True, overdraw: bool=True):
    BoardWriter(board, use_ascii).print(overdraw) 

north = Direction.north 
east = Direction.east 
south = Direction.south 
west = Direction.west 

board = generate_board(Vec2D(20, 20))
controller = TumbleController(board) # shuts up linter 

import numpy as np


def interpolant(t: Any):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape: tuple[int, int], res: tuple[int, int], tileable: tuple[bool, bool]=(False, False), 
        interpolant: Callable[[float], float]=interpolant
) -> Any:
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta: tuple[float, float] = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid) # type: ignore 
    n0: Any = n00*(1-t[:,:,0]) + t[:,:,0]*n10 # type: ignore 
    n1: Any = n01*(1-t[:,:,0]) + t[:,:,0]*n11 # type: ignore 
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) # type: ignore 


def generate_fractal_noise_2d(
        shape: tuple[int, int], res: tuple[int, int], octaves: int=1, persistence: float=0.5,
        lacunarity: int=2, tileable: tuple[bool, bool]=(False, False),
        interpolant: Callable[[float], float]=interpolant
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def get_image(path: str):
    im = cv2.imread(filename=path)
    if type(im) != np.ndarray: raise ValueError("Failed to load image.")

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im


def gray_conversion(image: Any):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


max_x = 512 
max_y = 512 
board_locs: list[list[Vec2D | None]] = []
for x in range(512):
    board_locs.append([])
    for y in range(512):
        board_locs[x].append(Vec2D(x, y)) 


def map_image_to_board(board: Board, image: Any):
    image = cv2.resize(image, dsize=(board.size.x, board.size.y), interpolation=cv2.INTER_CUBIC) # type: ignore 

    board.tiles.elements.clear() 
    board.concretes.clear() 

    for x in range(board.size.x):
        for y in range(board.size.y):
            cached = board_locs[x][y]
            if image[y][x] > 127: board.concretes.add(cached if cached else Vec2D(x, y))


class Stream():
    """
    extends [cv2::VideoCapture class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html)
    for video or stream subsampling.

    Parameters
    ----------
    filename : Union[str, int]
        Open video file or image file sequence or a capturing device
        or a IP video stream for video capturing.
    target_fps : int, optional
        the target frame rate. To ensure a constant time period between
        each subsampled frames, this parameter is used to compute a
        integer denominator for the extraction frequency. For instance,
        if the original stream is 64fps and you want a 30fps stream out,
        it is going to take one frame over two giving an effective frame
        rate of 32fps.
        If None, will extract every frame of the stream.
    """

    def __init__(self, filename: Union[str, int], target_fps: int | None = None):
        self.stream_id = filename
        self._cap = cv2.VideoCapture(self.stream_id)
        if not self.isOpened():
            raise FileNotFoundError("Stream not found")

        self.target_fps = target_fps
        self.fps = None
        self.extract_freq = None
        self.compute_extract_frequency()
        self._frame_index = 0

    def compute_extract_frequency(self):
        """evaluate the frame rate over a period of 5 seconds"""
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.compute_origin_fps()

        if self.target_fps is None: 
            self.extract_freq = 1
        else:
            self.extract_freq = int(self.fps / self.target_fps)

            if self.extract_freq == 0:
                raise ValueError("desired_fps is higher than half the stream frame rate")

    def compute_origin_fps(self, evaluation_period: int = 5):
        """evaluate the frame rate over a period of 5 seconds"""
        while self.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                if self._frame_index == 0:
                    start = time()

                self._frame_index += 1

                if time() - start > evaluation_period: # type: ignore 
                    break

        self.fps = round(self._frame_index / (time() - start), 2) # type: ignore 

    def read(self):
        """Grabs, decodes and returns the next subsampled video frame."""
        ret, frame = self._cap.read()
        if ret is True:
            self._frame_index += 1

            if self._frame_index == self.extract_freq:
                self._frame_index = 0
                return ret, frame

        return False, False

    def isOpened(self):
        """Returns true if video capturing has been initialized already."""
        return self._cap.isOpened()

    def release(self):
        """Closes video file or capturing device."""
        self._cap.release()


def play_video(board: Board, file: str):

    cap = cv2.VideoCapture(file)
    player = MediaPlayer(file) # type: ignore 
    start_time = time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, val = player.get_frame(show=False) # type: ignore 
        if val == 'eof':
            break

        cv2.imshow(file, frame)


        elapsed = (time() - start_time) * 1000  # msec
        play_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        sleep = max(1, int(play_time - elapsed))

        frame = gray_conversion(frame)
        map_image_to_board(board, frame)

        # print(chr(27) + "[2J")
        print_board(board, False)

        if cv2.waitKey(sleep) & 0xFF == ord("q"):
            break

    return 


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

    controller.tumble(Direction.east)
    writer.print()  

    controller.tumble(Direction.south)
    writer.print() 

    controller.tumble(Direction.west)
    writer.print() 

    controller.tumble(Direction.north)
    writer.print()

    controller.tumble(Direction.east)
    writer.print()  