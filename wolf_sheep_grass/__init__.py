import numpy as np
from attr import define, field


@define(kw_only=True)
class WolfSheepGrassModel:
    GRID_WIDTH: int = field()
    GRID_HEIGHT: int = field()

    MAX_WOLVES: int = field(default=10_000)
    MAX_SHEEP: int = field(default=10_000)

    INIT_WOLVES: int = field()  # 0..250
    WOLF_GAIN_FROM_FOOD: float = field()  # 0..100
    WOLF_REPRODUCE: float = field()  # 0..20

    INIT_SHEEP = field()  # 0..250
    SHEEP_GAIN_FROM_FOOD = field()  # 0..50
    SHEEP_REPRODUCE = field()  # 1..20

    INIT_GRASS_PROPORTION = field()  # 0..1
    GRASS_REGROWTH_TIME = field()  # 0..100

    grass: np.ndarray = field(init=False)
    grass_clock: np.ndarray = field(init=False)

    num_wolves: int = field(init=False)
    wolf_pos: np.ndarray = field(init=False)
    wolf_dir: np.ndarray = field(init=False)
    wolf_energy: np.ndarray = field(init=False)
    wolf_alive: np.ndarray = field(init=False)
    wolf_pointer: int = field(init=False)  # all indices >= this are not alive

    num_sheep: int = field(init=False)
    sheep_pos: np.ndarray = field(init=False)
    sheep_dir: np.ndarray = field(init=False)
    sheep_energy: np.ndarray = field(init=False)
    sheep_alive: np.ndarray = field(init=False)
    sheep_pointer: int = field(init=False)  # all indices >= this are not alive

    def __attrs_post_init__(self):
        self.num_wolves = 0
        self.wolf_pointer = 0
        self.wolf_pos = np.zeros((self.MAX_WOLVES, 2), dtype=np.float64)
        self.wolf_dir = np.zeros(self.MAX_WOLVES, dtype=np.float64)
        self.wolf_energy = np.zeros(self.MAX_WOLVES, dtype=np.float64)
        self.wolf_alive = np.zeros(self.MAX_WOLVES, dtype=bool)

        self.num_sheep = 0
        self.sheep_pointer = 0
        self.sheep_pos = np.zeros((self.MAX_SHEEP, 2), dtype=np.float64)
        self.sheep_dir = np.zeros(self.MAX_SHEEP, dtype=np.float64)
        self.sheep_energy = np.zeros(self.MAX_SHEEP, dtype=np.float64)
        self.sheep_alive = np.zeros(self.MAX_SHEEP, dtype=bool)

        self.grass = (
            np.random.rand(self.GRID_WIDTH, self.GRID_HEIGHT)
            < self.INIT_GRASS_PROPORTION
        )
        self.grass_clock = self.GRASS_REGROWTH_TIME * np.random.rand(
            self.GRID_WIDTH, self.GRID_HEIGHT
        )
        for _ in range(self.INIT_SHEEP):
            self.create_sheep()

        for _ in range(self.INIT_WOLVES):
            self.create_wolf()

    def compact_wolf_arrays(self):
        self.wolf_pos[: self.num_wolves] = self.wolf_pos[self.wolf_alive]
        self.wolf_dir[: self.num_wolves] = self.wolf_dir[self.wolf_alive]
        self.wolf_energy[: self.num_wolves] = self.wolf_energy[self.wolf_alive]
        self.wolf_alive[: self.num_wolves] = True
        self.wolf_alive[self.num_wolves :] = False
        self.wolf_pointer = self.num_wolves

    def compact_sheep_arrays(self):
        self.sheep_pos[: self.num_sheep] = self.sheep_pos[self.sheep_alive]
        self.sheep_dir[: self.num_sheep] = self.sheep_dir[self.sheep_alive]
        self.sheep_energy[: self.num_sheep] = self.sheep_energy[self.sheep_alive]
        self.sheep_alive[: self.num_sheep] = True
        self.sheep_alive[self.num_sheep :] = False
        self.sheep_pointer = self.num_sheep

    def create_wolf(self, pos=None, energy=None):
        if self.wolf_pointer >= self.MAX_WOLVES:
            self.compact_wolf_arrays()
            # maybe the array is already compacted:
            if self.wolf_pointer >= self.MAX_WOLVES:
                raise RuntimeError(
                    "Max wolves exceeded, you may want to change the MAX_WOLVES parameter."
                )
        if pos is None:
            self.wolf_pos[self.wolf_pointer, 0] = self.GRID_WIDTH * np.random.rand()
            self.wolf_pos[self.wolf_pointer, 1] = self.GRID_HEIGHT * np.random.rand()
        else:
            self.wolf_pos[self.wolf_pointer] = pos
        self.wolf_dir[self.wolf_pointer] = 2 * np.pi * np.random.rand()
        if energy is None:
            self.wolf_energy[self.wolf_pointer] = (
                2 * self.WOLF_GAIN_FROM_FOOD * np.random.rand()
            )
        else:
            self.wolf_energy[self.wolf_pointer] = energy
        self.wolf_alive[self.wolf_pointer] = True
        self.num_wolves += 1
        self.wolf_pointer += 1

    def create_sheep(self, pos=None, energy=None):
        if self.sheep_pointer >= self.MAX_SHEEP:
            self.compact_sheep_arrays()
            # maybe the array is already compacted:
            if self.sheep_pointer >= self.MAX_SHEEP:
                raise RuntimeError(
                    "Max sheep exceeded, you may want to change the MAX_SHEEP parameter."
                )
        if pos is None:
            self.sheep_pos[self.sheep_pointer, 0] = self.GRID_WIDTH * np.random.rand()
            self.sheep_pos[self.sheep_pointer, 1] = self.GRID_HEIGHT * np.random.rand()
        else:
            self.sheep_pos[self.sheep_pointer] = pos
        self.sheep_dir[self.sheep_pointer] = 2 * np.pi * np.random.rand()
        if energy is None:
            self.sheep_energy[self.sheep_pointer] = (
                2 * self.SHEEP_GAIN_FROM_FOOD * np.random.rand()
            )
        else:
            self.sheep_energy[self.sheep_pointer] = energy
        self.sheep_alive[self.sheep_pointer] = True
        self.num_sheep += 1
        self.sheep_pointer += 1

    def sheep_move(self):
        self.sheep_dir += (2 * np.random.rand(self.MAX_SHEEP) - 1) * 2 * np.pi / 50
        directions = np.stack([np.cos(self.sheep_dir), np.sin(self.sheep_dir)], axis=1)
        self.sheep_pos += directions
        self.sheep_pos[:, 0] = self.sheep_pos[:, 0] % self.GRID_WIDTH
        self.sheep_pos[:, 1] = self.sheep_pos[:, 1] % self.GRID_HEIGHT

    def wolves_move(self):
        self.wolf_dir += (2 * np.random.rand(self.MAX_WOLVES) - 1) * 2 * np.pi / 50
        directions = np.stack([np.cos(self.wolf_dir), np.sin(self.wolf_dir)], axis=1)
        self.wolf_pos += directions
        self.wolf_pos[:, 0] = self.wolf_pos[:, 0] % self.GRID_WIDTH
        self.wolf_pos[:, 1] = self.wolf_pos[:, 1] % self.GRID_HEIGHT

    def sheep_eat_grass(self):
        for idx in range(self.sheep_pointer):
            if not self.sheep_alive[idx]:
                continue
            x = int(self.sheep_pos[idx][0])
            y = int(self.sheep_pos[idx][1])
            if self.grass[x, y]:
                self.sheep_energy[idx] += self.SHEEP_GAIN_FROM_FOOD
                self.grass[x, y] = False
                self.grass_clock[x, y] = self.GRASS_REGROWTH_TIME

    def wolves_eat_sheep(self):
        sheep_locs = np.int64(self.sheep_pos)
        for idx in np.where(self.wolf_alive)[0]:
            x = int(self.wolf_pos[idx][0])
            y = int(self.wolf_pos[idx][1])
            # find all (alive) self.sheep in the same 'pixel'
            local_sheep_idcs = np.where(
                np.logical_and(
                    self.sheep_alive,
                    np.logical_and(sheep_locs[:, 0] == x, sheep_locs[:, 1] == y),
                )
            )[0]
            num_local_sheep = local_sheep_idcs.shape[0]
            if num_local_sheep <= 0:
                continue
            # select one at random and eat it
            local_sheep_idx = local_sheep_idcs[np.random.randint(0, num_local_sheep)]
            self.sheep_alive[local_sheep_idx] = False
            self.wolf_energy[idx] += self.WOLF_GAIN_FROM_FOOD
            self.num_sheep -= 1

    def sheep_die(self):
        np.logical_and(self.sheep_alive, self.sheep_energy >= 0.0, out=self.sheep_alive)
        self.num_sheep = int(np.sum(self.sheep_alive))
        live_sheep = np.where(self.sheep_alive)[0]
        if live_sheep.shape[0] == 0:
            self.sheep_pointer = 0
        else:
            self.sheep_pointer = live_sheep[-1] + 1

    def wolves_die(self):
        np.logical_and(self.wolf_alive, self.wolf_energy >= 0.0, out=self.wolf_alive)
        self.num_wolves = int(np.sum(self.wolf_alive))
        live_wolves = np.where(self.wolf_alive)[0]
        if live_wolves.shape[0] == 0:
            self.wolf_pointer = 0
        else:
            self.wolf_pointer = live_wolves[-1] + 1

    def sheep_reproduce(self):
        reproduce = np.logical_and(
            self.sheep_alive,
            np.random.rand(self.MAX_SHEEP) < self.SHEEP_REPRODUCE / 100.0,
        )
        self.sheep_energy[reproduce] /= 2.0
        reproducing_sheep_pos = np.copy(self.sheep_pos[reproduce])
        reproducing_sheep_energy = np.copy(self.sheep_energy[reproduce])

        for idx in range(np.sum(reproduce)):
            self.create_sheep(
                pos=reproducing_sheep_pos[idx], energy=reproducing_sheep_energy[idx]
            )

    def wolves_reproduce(self):
        reproduce = np.logical_and(
            self.wolf_alive,
            np.random.rand(self.MAX_WOLVES) < self.WOLF_REPRODUCE / 100.0,
        )
        self.wolf_energy[reproduce] /= 2.0
        reproducing_wolf_pos = np.copy(self.wolf_pos[reproduce])
        reproducing_wolf_energy = np.copy(self.wolf_energy[reproduce])

        for idx in range(np.sum(reproduce)):
            self.create_wolf(
                pos=reproducing_wolf_pos[idx], energy=reproducing_wolf_energy[idx]
            )

    def grow_grass(self):
        self.grass_clock -= 1
        self.grass_clock[self.grass] = 0
        self.grass[:] = self.grass_clock <= 0

    def time_step(self):
        # sheep
        self.sheep_move()
        self.sheep_energy -= 1.0  # self.sheep metabolism
        self.sheep_eat_grass()
        self.sheep_die()
        self.sheep_reproduce()

        # wolves
        self.wolves_move()
        self.wolf_energy -= 1.0  # wolf metabolism
        self.wolves_eat_sheep()
        self.wolves_die()
        self.wolves_reproduce()

        # grass
        self.grow_grass()
