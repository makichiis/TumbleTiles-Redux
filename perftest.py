import copy
from typing import NamedTuple 
from tilt.board import Singleton, Vec2D

class Info(object):
    __slots__ = ['x', 'y', 'a', 'b', 'c']

    def swap(self):
        newObj = copy.copy(self)
        newObj.x, newObj.y = self.y, self.x
        return newObj

    # for the sake of convenience
    def __init__(self, x: int, y: int, a: int, b: int, c: int):
        self.x = x
        self.y = y

class BaseInfo(NamedTuple):
    x: int 
    y: int 
    a: int 
    b: int 
    c: int 

class TupleInfo(BaseInfo):
    __slots__ = ()

    def swap(self):
        return self._replace(x=self.y, y=self.x)

if __name__ == "__main__":
    from timeit import timeit

    i1 = Info(1, 2, 0, 0, 0)
    i2 = TupleInfo(1, 2, 0, 0, 0)

    tile1 = Singleton(pos=Vec2D(5, 2))

    xs: dict[int, int] = {}
    sxs: list[int] = list()
    ys: set[int] = set() 

    for i in range(1000000):
        xs[i] = 2*i 
        sxs.append(i)
        ys.add(i)

    print("Built from scratch")
    print(timeit("z = i1.swap()", "from __main__ import i1", number=100000))

    print("Derived from namedtuple")
    print(timeit("z = i2.swap()", "from __main__ import i2", number=100000))

    print("Tile permutation")
    print(timeit("tile1 = Singleton(pos=Vec2D(tile1.pos.x + 1, tile1.pos.y + 1))", "from __main__ import tile1\nfrom tilt.board import Singleton, Vec2D", number=100000))

    print("Tile mutation")
    print(timeit("tile1.pos = Vec2D(tile1.pos.x + 1, tile1.pos.y + 1)", "from __main__ import tile1\nfrom tilt.board import Vec2D", number=100000))

    print("Traversing list")
    print(timeit("for x in sxs: pass", "from __main__ import sxs", number=100))

    print("Traversing dict")
    print(timeit("for x in xs: pass", "from __main__ import xs", number=100))

    print("Traversing set")
    print(timeit("for y in ys: pass", "from __main__ import ys", number=100))
