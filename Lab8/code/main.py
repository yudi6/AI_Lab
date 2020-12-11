from collections import deque


class Search:
    def __init__(self, search_policy):
        # 起点 终点 图 图的长与宽（在本次实验中为18*36）
        self.start, self.end, self.map, self.map_length, self.map_width = self._load_map()
        # 采用的搜索策略
        self.search_policy = search_policy
        # 在节点的行动到下标的映射
        self.action_to_ix = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
        # 搜索到的最终路径图
        self.result_map = self.map
        # 每个节点的实际代价 g(x)
        self.point_g = {}

    def begin_search(self):
        if self.search_policy == 'UCS':
            self._UCS()
        elif self.search_policy == 'Astar':
            self._Astar()

    def _can_reach(self, state, action):
        # Up
        if action == 0:
            return (state[0] - 1, state[1]), self.map[(state[0] - 1, state[1])] != '1'
        # Down
        elif action == 1:
            return (state[0] + 1, state[1]), self.map[(state[0] + 1, state[1])] != '1'
        # Left
        elif action == 2:
            return (state[0], state[1] - 1), self.map[(state[0], state[1] - 1)] != '1'
        # Right
        else:
            return (state[0], state[1] + 1), self.map[(state[0], state[1] + 1)] != '1'

    def _get_g(self, point):
        return self.point_g[point]

    def _UCS(self):
        count = 0                   # 搜索次数
        father_point = {}           # 记录父节点
        queue_length = []
        search_queue = []       # 搜索队列
        search_queue.append(self.start)
        self.point_g[self.start] = 0        # 初始化起点的真实代价
        visted = []                 # 保存已搜索过的节点
        flag = True
        while flag and search_queue:
            queue_length.append(len(search_queue))
            point = min(search_queue, key=self._get_g)
            search_queue.remove(point)
            count += 1
            # 判断是否到达终点
            if self._gold_test(point):
                flag = False
                break
            # 相邻的可到达节点入队
            for action in range(4):
                next, reachable = self._can_reach(point, action)
                if next not in visted and reachable:
                    search_queue.append(next)
                    self.point_g[next] = self.point_g[point] + 1
                    # 保存父节点
                    if next not in father_point:
                        father_point[next] = point
            visted.append(point)
        # 输出结果
        if not flag:
            print('Using', self.search_policy, 'needs', count, 'steps')
            print('Using', self.search_policy, 'needs', max(queue_length), 'queue length')
            self._get_result_map(father_point)
            self._print_map(self.result_map)
        else:
            print('No result')

    def _heuristic_plus(self, point):
        return abs(point[0] - self.end[0]) + abs(point[1] - self.end[1]) + self.point_g[point]

    def _Astar(self):
        count = 0               # 搜索次数
        father_point = {}       # 记录父节点
        queue_length = []
        search_queue = []       # 搜索队列
        search_queue.append(self.start)
        self.point_g[self.start] = 0        # 初始化起点的真实代价
        visted = []             # 保存已搜索过的节点
        flag = True
        while flag and search_queue:
            queue_length.append(len(search_queue))
            # 选择最小估值函数的扩展节点
            point = min(search_queue, key=self._heuristic_plus)
            search_queue.remove(point)
            if self._gold_test(point):
                flag = False
                break
            for action in range(4):
                next, reachable = self._can_reach(point, action)
                # 判断是否更新真实代价
                if next in search_queue and self.point_g[point] + 1 < self.point_g[next]:
                    self.point_g[next] = self.point_g[point] + 1
                    continue
                # 相邻节点入队
                if next not in visted and reachable:
                    search_queue.append(next)
                    # 保存真实代价
                    self.point_g[next] = self.point_g[point] + 1
                    if next not in father_point:
                        father_point[next] = point
            visted.append(point)
            count += 1
        # 打印结果
        if not flag:
            print('Using', self.search_policy, 'needs', count, 'steps')
            print('Using', self.search_policy, 'needs', max(queue_length), 'queue length')
            self._get_result_map(father_point)
            self._print_map(self.result_map)
        else:
            print('No result')

    def _get_result_map(self, father_point):
        count = 1
        step = father_point[self.end]
        while step != self.start:
            self.result_map[step] = '·'
            step = father_point[step]
            count += 1
        print('The path length found is', count)

    def _print_map(self, map_to_print):
        map_list = []
        for i in range(self.map_length):
            line = [map_to_print[(i, j)] for j in range(self.map_width)]
            map_list.append("".join(line))
        for line in map_list:
            print(line)

    def _load_map(self):
        # 采用字典保存读取的图
        map = {}
        with open('MazeData.txt', 'r') as fp:
            map_list = fp.readlines()
        for i in range(len(map_list)):
            for j in range(len(map_list[i]) - 1):
                # 对应（i, j) 位置的字符保存
                map[(i, j)] = map_list[i][j]
                # 保存起点
                if map[(i, j)] == 'S':
                    start = (i, j)
                # 保存终点
                if map[(i, j)] == 'E':
                    end = (i, j)
        return start, end, map, len(map_list), len(map_list[0]) - 1

    def _gold_test(self, point):
        return point == self.end


if __name__ == '__main__':
    search_A = Search('UCS')
    search_A.begin_search()
    print('')
    search_B = Search('Astar')
    search_B.begin_search()
