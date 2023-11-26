import numpy as np
import matplotlib.pyplot as plt


class Maze:
    def __init__(self, *segment_dicts, goal_squares=None, start_squares=None):
        self._segments = {'origin': {'loc': (0.0, 0.0), 'connect': set()}}
        self._locs = set()
        self._locs.add(self._segments['origin']['loc'])
        self._walls = set()
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(
                self._segments['origin']['loc'], direction
            ))
        self._last_segment = 'origin'
        self.goal_squares = None

        if goal_squares is None:
            self._goal_squares = None
        elif isinstance(goal_squares, str):
            self._goal_squares = [goal_squares.lower()]
        elif isinstance(goal_squares, (tuple, list)):
            self._goal_squares = [gs.lower() for gs in goal_squares]
        else:
            raise TypeError

        if start_squares is None:
            self.start_squares = ['origin']
        elif isinstance(start_squares, str):
            self.start_squares = [start_squares.lower()]
        elif isinstance(start_squares, (tuple, list)):
            self.start_squares = [ss.lower() for ss in start_squares]
        else:
            raise TypeError

        for segment_dict in segment_dicts:
            self._add_segment(**segment_dict)
        self._finalize()
        self.seed()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    @staticmethod
    def _wall_line(coord, direction):
        x, y = coord
        if direction == 'up':
            w = [(x - 0.5, x + 0.5), (y + 0.5, y + 0.5)]
        elif direction == 'right':
            w = [(x + 0.5, x + 0.5), (y + 0.5, y - 0.5)]
        elif direction == 'down':
            w = [(x - 0.5, x + 0.5), (y - 0.5, y - 0.5)]
        elif direction == 'left':
            w = [(x - 0.5, x - 0.5), (y - 0.5, y + 0.5)]
        else:
            raise ValueError
        w = tuple([tuple(sorted(line)) for line in w])
        return w

    def _add_segment(self, name, anchor, direction, connect=None, times=1):
        name = str(name).lower()
        original_name = str(name).lower()
        if times > 1:
            assert connect is None
            last_name = str(anchor).lower()
            for time in range(times):
                this_name = original_name + str(time)
                self._add_segment(
                    name=this_name.lower(), anchor=last_name,
                    direction=direction
                )
                last_name = str(this_name)
            return

        anchor = str(anchor).lower()
        assert anchor in self._segments

        direction = str(direction).lower()

        final_connect = set()

        if connect is not None:
            if isinstance(connect, str):
                connect = str(connect).lower()
                assert connect in ['up', 'down', 'left', 'right']
                final_connect.add(connect)
            elif isinstance(connect, (tuple, list)):
                for connect_direction in connect:
                    connect_direction = str(connect_direction).lower()
                    assert connect_direction in ['up', 'down', 'left', 'right']
                    final_connect.add(connect_direction)

        sx, sy = self._segments[anchor]['loc']
        dx, dy = 0.0, 0.0
        if direction == 'left':
            dx -= 1
            final_connect.add('right')
        elif direction == 'right':
            dx += 1
            final_connect.add('left')
        elif direction == 'up':
            dy += 1
            final_connect.add('down')
        elif direction == 'down':
            dy -= 1
            final_connect.add('up')
        else:
            raise ValueError

        new_loc = (sx + dx, sy + dy)
        assert new_loc not in self._locs

        self._segments[name] = {'loc': new_loc, 'connect': final_connect}
        for direction in ['up', 'down', 'left', 'right']:
            self._walls.add(self._wall_line(new_loc, direction))
        self._locs.add(new_loc)

        self._last_segment = name

    def _finalize(self):
        for segment in self._segments.values():
            for c_dir in list(segment['connect']):
                wall = self._wall_line(segment['loc'], c_dir)
                if wall in self._walls:
                    self._walls.remove(wall)

        if self._goal_squares is None:
            self.goal_squares = [self._last_segment]
        else:
            self.goal_squares = []
            for gs in self._goal_squares:
                assert gs in self._segments
                self.goal_squares.append(gs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(5, 4))
        for x, y in self._walls:
            ax.plot(x, y, 'k-', zorder=10)

    def sample_start(self):
        min_wall_dist = 0.05

        s_square = self.start_squares[self.np_random.integers(
            low=0, high=len(self.start_squares)
        )]
        s_square_loc = self._segments[s_square]['loc']

        while True:
            shift = self.np_random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = s_square_loc + shift
            dist_checker = np.array(
                [min_wall_dist, min_wall_dist]
            ) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def sample_goal(self, min_wall_dist=None, goal_idx=None):
        if min_wall_dist is None:
            min_wall_dist = 0.1
        else:
            min_wall_dist = min(0.4, max(0.01, min_wall_dist))
        if goal_idx is None:
            g_square = self.goal_squares[self.np_random.integers(
                low=0, high=len(self.goal_squares)
            )]
        else:
            g_square = self.goal_squares[goal_idx]
        g_square_loc = self._segments[g_square]['loc']
        while True:
            shift = self.np_random.uniform(low=-0.5, high=0.5, size=(2,))
            loc = g_square_loc + shift
            dist_checker = np.array(
                [min_wall_dist, min_wall_dist]
            ) * np.sign(shift)
            stopped_loc = self.move(loc, dist_checker)
            if float(np.sum(np.abs((loc + dist_checker) - stopped_loc))) == 0.0:
                break
        return loc[0], loc[1]

    def move(self, coord_start, coord_delta, depth=None):
        if depth is None:
            depth = 0
        cx, cy = coord_start
        loc_x0 = np.round(cx)
        loc_y0 = np.round(cy)
        dx, dy = coord_delta
        loc_x1 = np.round(cx + dx)
        loc_y1 = np.round(cy + dy)
        d_loc_x = int(np.abs(loc_x1 - loc_x0))
        d_loc_y = int(np.abs(loc_y1 - loc_y0))
        xs_crossed = [loc_x0 + (np.sign(dx) * (i + 0.5))
                      for i in range(d_loc_x)]
        ys_crossed = [loc_y0 + (np.sign(dy) * (i + 0.5))
                      for i in range(d_loc_y)]

        rds = []

        for x in xs_crossed:
            r = (x - cx) / dx
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'right' if dx > 0 else 'left'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        for y in ys_crossed:
            r = (y - cy) / dy
            loc_x = np.round(cx + (0.999 * r * dx))
            loc_y = np.round(cy + (0.999 * r * dy))
            direction = 'up' if dy > 0 else 'down'
            crossed_line = self._wall_line((loc_x, loc_y), direction)
            if crossed_line in self._walls:
                rds.append([r, direction])

        # The wall will only stop the agent in the direction perpendicular
        # to the wall
        if rds:
            rds = sorted(rds)
            r, direction = rds[0]
            if depth < 3:
                new_dx = r * dx
                new_dy = r * dy
                repulsion = float(np.abs(self.np_random.random() * 0.01))
                if direction in ['right', 'left']:
                    new_dx -= np.sign(dx) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = (0.0, (1 - r) * dy)
                else:
                    new_dy -= np.sign(dy) * repulsion
                    partial_coords = cx + new_dx, cy + new_dy
                    remaining_delta = ((1 - r) * dx, 0.0)
                return self.move(partial_coords, remaining_delta, depth+1)
        else:
            r = 1.0

        dx *= r
        dy *= r
        return cx + dx, cy + dy


mazes_dict = dict()

segments_crazy = [
    {'anchor': 'origin', 'direction': 'right', 'name': '1,0'},
    {'anchor': 'origin', 'direction': 'up', 'name': '0,1'},
    {'anchor': '1,0', 'direction': 'right', 'name': '2,0'},
    {'anchor': '0,1', 'direction': 'up', 'name': '0,2'},
    {'anchor': '0,2', 'direction': 'right', 'name': '1,2'},
    {'anchor': '2,0', 'direction': 'up', 'name': '2,1'},
    {'anchor': '1,2', 'direction': 'right', 'name': '2,2'},
    {'anchor': '0,2', 'direction': 'up', 'name': '0,3'},
    {'anchor': '2,1', 'direction': 'right', 'name': '3,1'},
    {'anchor': '1,2', 'direction': 'down', 'name': '1,1'},
    {'anchor': '3,1', 'direction': 'down', 'name': '3,0'},
    {'anchor': '1,2', 'direction': 'up', 'name': '1,3'},
    {'anchor': '3,1', 'direction': 'right', 'name': '4,1'},
    {'anchor': '1,3', 'direction': 'up', 'name': '1,4'},
    {'anchor': '4,1', 'direction': 'right', 'name': '5,1'},
    {'anchor': '4,1', 'direction': 'up', 'name': '4,2'},
    {'anchor': '5,1', 'direction': 'down', 'name': '5,0'},
    {'anchor': '3,0', 'direction': 'right', 'name': '4,0'},
    {'anchor': '1,4', 'direction': 'right', 'name': '2,4'},
    {'anchor': '4,2', 'direction': 'right', 'name': '5,2'},
    {'anchor': '2,4', 'direction': 'right', 'name': '3,4'},
    {'anchor': '3,4', 'direction': 'up', 'name': '3,5'},
    {'anchor': '1,4', 'direction': 'left', 'name': '0,4'},
    {'anchor': '1,4', 'direction': 'up', 'name': '1,5'},
    {'anchor': '2,2', 'direction': 'up', 'name': '2,3'},
    {'anchor': '3,1', 'direction': 'up', 'name': '3,2'},
    {'anchor': '5,0', 'direction': 'right', 'name': '6,0'},
    {'anchor': '3,2', 'direction': 'up', 'name': '3,3'},
    {'anchor': '4,2', 'direction': 'up', 'name': '4,3'},
    {'anchor': '6,0', 'direction': 'up', 'name': '6,1'},
    {'anchor': '6,0', 'direction': 'right', 'name': '7,0'},
    {'anchor': '6,1', 'direction': 'right', 'name': '7,1'},
    {'anchor': '3,4', 'direction': 'right', 'name': '4,4'},
    {'anchor': '1,5', 'direction': 'right', 'name': '2,5'},
    {'anchor': '7,1', 'direction': 'up', 'name': '7,2'},
    {'anchor': '1,5', 'direction': 'up', 'name': '1,6'},
    {'anchor': '4,4', 'direction': 'right', 'name': '5,4'},
    {'anchor': '5,4', 'direction': 'down', 'name': '5,3'},
    {'anchor': '0,4', 'direction': 'up', 'name': '0,5'},
    {'anchor': '7,2', 'direction': 'left', 'name': '6,2'},
    {'anchor': '1,6', 'direction': 'left', 'name': '0,6'},
    {'anchor': '7,0', 'direction': 'right', 'name': '8,0'},
    {'anchor': '7,2', 'direction': 'right', 'name': '8,2'},
    {'anchor': '2,5', 'direction': 'up', 'name': '2,6'},
    {'anchor': '8,0', 'direction': 'up', 'name': '8,1'},
    {'anchor': '3,5', 'direction': 'up', 'name': '3,6'},
    {'anchor': '6,2', 'direction': 'up', 'name': '6,3'},
    {'anchor': '6,3', 'direction': 'right', 'name': '7,3'},
    {'anchor': '3,5', 'direction': 'right', 'name': '4,5'},
    {'anchor': '7,3', 'direction': 'up', 'name': '7,4'},
    {'anchor': '6,3', 'direction': 'up', 'name': '6,4'},
    {'anchor': '6,4', 'direction': 'up', 'name': '6,5'},
    {'anchor': '8,1', 'direction': 'right', 'name': '9,1'},
    {'anchor': '8,2', 'direction': 'right', 'name': '9,2'},
    {'anchor': '2,6', 'direction': 'up', 'name': '2,7'},
    {'anchor': '8,2', 'direction': 'up', 'name': '8,3'},
    {'anchor': '6,5', 'direction': 'left', 'name': '5,5'},
    {'anchor': '5,5', 'direction': 'up', 'name': '5,6'},
    {'anchor': '7,4', 'direction': 'right', 'name': '8,4'},
    {'anchor': '8,4', 'direction': 'right', 'name': '9,4'},
    {'anchor': '0,6', 'direction': 'up', 'name': '0,7'},
    {'anchor': '2,7', 'direction': 'up', 'name': '2,8'},
    {'anchor': '7,4', 'direction': 'up', 'name': '7,5'},
    {'anchor': '9,4', 'direction': 'down', 'name': '9,3'},
    {'anchor': '9,4', 'direction': 'up', 'name': '9,5'},
    {'anchor': '2,7', 'direction': 'left', 'name': '1,7'},
    {'anchor': '4,5', 'direction': 'up', 'name': '4,6'},
    {'anchor': '9,1', 'direction': 'down', 'name': '9,0'},
    {'anchor': '6,5', 'direction': 'up', 'name': '6,6'},
    {'anchor': '3,6', 'direction': 'up', 'name': '3,7'},
    {'anchor': '1,7', 'direction': 'up', 'name': '1,8'},
    {'anchor': '3,7', 'direction': 'right', 'name': '4,7'},
    {'anchor': '2,8', 'direction': 'up', 'name': '2,9'},
    {'anchor': '2,9', 'direction': 'left', 'name': '1,9'},
    {'anchor': '7,5', 'direction': 'up', 'name': '7,6'},
    {'anchor': '1,8', 'direction': 'left', 'name': '0,8'},
    {'anchor': '6,6', 'direction': 'up', 'name': '6,7'},
    {'anchor': '0,8', 'direction': 'up', 'name': '0,9'},
    {'anchor': '7,5', 'direction': 'right', 'name': '8,5'},
    {'anchor': '6,7', 'direction': 'left', 'name': '5,7'},
    {'anchor': '2,9', 'direction': 'right', 'name': '3,9'},
    {'anchor': '3,9', 'direction': 'right', 'name': '4,9'},
    {'anchor': '7,6', 'direction': 'right', 'name': '8,6'},
    {'anchor': '3,7', 'direction': 'up', 'name': '3,8'},
    {'anchor': '9,5', 'direction': 'up', 'name': '9,6'},
    {'anchor': '7,6', 'direction': 'up', 'name': '7,7'},
    {'anchor': '5,7', 'direction': 'up', 'name': '5,8'},
    {'anchor': '3,8', 'direction': 'right', 'name': '4,8'},
    {'anchor': '8,6', 'direction': 'up', 'name': '8,7'},
    {'anchor': '5,8', 'direction': 'right', 'name': '6,8'},
    {'anchor': '7,7', 'direction': 'up', 'name': '7,8'},
    {'anchor': '4,9', 'direction': 'right', 'name': '5,9'},
    {'anchor': '8,7', 'direction': 'right', 'name': '9,7'},
    {'anchor': '7,8', 'direction': 'right', 'name': '8,8'},
    {'anchor': '8,8', 'direction': 'up', 'name': '8,9'},
    {'anchor': '5,9', 'direction': 'right', 'name': '6,9'},
    {'anchor': '6,9', 'direction': 'right', 'name': '7,9'},
    {'anchor': '8,9', 'direction': 'right', 'name': '9,9'},
    {'anchor': '9,9', 'direction': 'down', 'name': '9,8'}
]

# empty = [
#     {'anchor': 'origin', 'direction': 'right', 'name': '1,0'},
#     {'anchor': 'origin', 'direction': 'up', 'name': '0,1'}
# ]

# for i in range(9):
#     for j in range(9):
#         if i == 0 and j == 0:
#             continue
#         empty.append(
#             {'anchor': f'{i},{j}', 'direction': 'up', 'name': f'{i+1},{j}'}
#         )
#         empty.append(
#             {'anchor': f'{i},{j}', 'direction': 'right', 'name': f'{i},{j+1}'}
#         )

# for i in range(len(segments_crazy)):
#     if segments_crazy[i]['name'][0] not in ['0', '9']\
#         and segments_crazy[i]['name'][2] not in ['0', '9']:
#         segments_crazy[i]['connect'] = ['up', 'down', 'right', 'left']

# Square Large
mazes_dict['square_large'] = {
    'maze': Maze(*segments_crazy, goal_squares='9,9'),
    'action_range': 0.95
}

# MultiGoal Square Large
square_large_goals = [
    '9,9', '9,4', '9,0', '6,9', '6,4', '6,0', '3,9', '3,4', '3,0', '0,9', '0,4'
]
mazes_dict['multigoal_square_large'] = {
    'maze': Maze(*segments_crazy, goal_squares=square_large_goals),
    'action_range': 0.95
}
