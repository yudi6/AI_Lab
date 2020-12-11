from graphics import *


class GameTree():
    def __init__(self, first):
        self.first = first              # 谁先手
        self.column = 11                # 棋盘列数
        self.row = 11                   # 棋盘行数
        self.grid_width = 40            # UI界面格子的大小
        self.chessboard = self._init_chessboard()       # 初始化棋盘
        self.AI_pieces = [(5,5),(6,5)]if first == 'AI' else [(5,6),(4,5)]              # 根据先后手初始化AI的棋子
        self.player_pieces = [(5,5),(6,5)]if not first == 'AI' else [(5,6),(4,5)]      # 初始化玩家的棋子
        # self.AI_pieces = []
        # self.player_pieces = []
        # 打印初始化的棋子
        for point in self.AI_pieces:
            self._print_piece(point,'white')
        for point in self.player_pieces:
            self._print_piece(point, 'black')
        self.pos_in_chessboard = [  (i,j)  for i in range(self.column) for j in range(self.row)]    # 初始化棋盘的所有位置
        # 进攻棋形的相应得分，评价函数用
        self.shape_score = [(50, (0, 1, 1, 0, 0)),
               (50, (0, 0, 1, 1, 0)),
               (200, (1, 1, 0, 1, 0)),
               (500, (0, 0, 1, 1, 1)),
               (500, (1, 1, 1, 0, 0)),
               (5000, (0, 1, 1, 1, 0)),
               (5000, (0, 1, 0, 1, 1, 0)),
               (5000, (0, 1, 1, 0, 1, 0)),
               (5000, (1, 1, 1, 0, 1)),
               (5000, (1, 1, 0, 1, 1)),
               (5000, (1, 0, 1, 1, 1)),
               (5000, (1, 1, 1, 1, 0)),
               (5000, (0, 1, 1, 1, 1)),
               (50000, (0, 1, 1, 1, 1, 0)),
               (99999999, (1, 1, 1, 1, 1))]
        self._play()
    def _print_score(self):
        r = Rectangle(Point(0, 0), Point(300, 40))
        r.setFill('yellow')
        r.draw(self.chessboard)
        Text(r.getCenter(),'Score is '+str(self._eva())).draw(self.chessboard)

    def _init_chessboard(self):
        win = GraphWin("五子棋", self.grid_width * (self.column+2), self.grid_width * (self.row+2))
        win.setBackground("yellow")
        point_on_line = 3*self.grid_width/2
        # 画线
        while point_on_line < self.grid_width * (self.column+1):
            l = Line(Point(point_on_line, 3*self.grid_width/2), Point(point_on_line, self.grid_width * (self.column-1)+3*self.grid_width/2))
            l.draw(win)
            point_on_line = point_on_line + self.grid_width
        point_on_line = 3*self.grid_width/2
        while point_on_line < self.grid_width * (self.row+1):
            l = Line(Point(3*self.grid_width/2, point_on_line), Point(self.grid_width * (self.row-1)+3*self.grid_width/2, point_on_line))
            l.draw(win)
            point_on_line = point_on_line + self.grid_width
        return win

    def _print_piece(self, pos, color):
        piece = Circle(Point(self.grid_width * pos[0]+3*self.grid_width/2, self.grid_width * pos[1]+3*self.grid_width/2), 16)
        piece.setFill(color)
        piece.draw(self.chessboard)

    def _win(self, pieces_list):
        for pos in pieces_list:
            x = pos[0]
            y = pos[1]
            if y < self.row - 4 and (x, y + 1) in pieces_list and (x, y + 2) in pieces_list and (
                        x, y + 3) in pieces_list and (x, y + 4) in pieces_list:
                return True
            elif x < self.row - 4 and (x + 1, y) in pieces_list and (x + 2, y) in pieces_list and (
                        x + 3, y) in pieces_list and (x + 4, y) in pieces_list:
                return True
            elif x < self.row - 4 and y < self.row - 4 and (x + 1, y + 1) in pieces_list and (
                        x + 2, y + 2) in pieces_list and (x + 3, y + 3) in pieces_list and (x + 4, y + 4) in pieces_list:
                return True
            elif x < self.row - 4 and y > 3 and (x + 1, y - 1) in pieces_list and (
                        x + 2, y - 2) in pieces_list and (x + 3, y - 3) in pieces_list and (x + 4, y - 4) in pieces_list:
                return True
        return False

    def _eva(self):
        # 玩家的得分棋阵
        player_score_list = [] 
        player_score = 0
        # 每个棋子在四个方向根据棋形计算
        for piece in self.player_pieces:
            x = piece[0]
            y = piece[1]
            player_score += self._cal_score(x, y, 0, 1, self.AI_pieces, self.player_pieces, player_score_list)
            player_score += self._cal_score(x, y, 1, 0, self.AI_pieces, self.player_pieces, player_score_list)
            player_score += self._cal_score(x, y, 1, 1, self.AI_pieces, self.player_pieces, player_score_list)
            player_score += self._cal_score(x, y, -1, 1, self.AI_pieces, self.player_pieces, player_score_list)
        # AI的得分棋阵
        AI_score_list = []
        AI_score = 0
        # 每个棋子在四个方向根据棋形计算
        for piece in self.AI_pieces:
            x = piece[0]
            y = piece[1]
            AI_score += self._cal_score(x, y, 0, 1, self.player_pieces, self.AI_pieces, AI_score_list)
            AI_score += self._cal_score(x, y, 1, 0, self.player_pieces, self.AI_pieces, AI_score_list)
            AI_score += self._cal_score(x, y, 1, 1, self.player_pieces, self.AI_pieces, AI_score_list)
            AI_score += self._cal_score(x, y, -1, 1, self.player_pieces, self.AI_pieces, AI_score_list)
        # 最后棋盘的评估
        return player_score - AI_score

    def _cal_score(self, x, y, dx, dy, enemy_list, my_list, my_score_list):
        add_score = 0
        # 在已确定的方向选择分数最大的形状
        max_score_shape = (0, None)
        # 在同个方向已计算过的则跳过 防止重复计算
        for score_piece in my_score_list:
            for pt in score_piece[1]:
                if (x, y) == pt and (dx, dy) == score_piece[2]:
                    return 0
                
        # 在该方向上得到当前局势的棋形
        for offset in range(-5, 1):
            pos = []
            for i in range(0, 6):
                pos_now = (x + (i + offset) * dx, y + (i + offset) * dy)
                if pos_now in enemy_list:
                    pos.append(2)
                elif pos_now in my_list:
                    pos.append(1)
                else:
                    pos.append(0)
            # 当前棋形
            shape_5 = (pos[0], pos[1], pos[2], pos[3], pos[4])
            shape_6 = (pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

            for (score, shape) in self.shape_score:
                if shape_5 == shape or shape_6 == shape:
                    # 判断是否为当前最大得分的棋形
                    if score > max_score_shape[0]:
                        # 保存最大得分的（得分，（各棋子位置），方向）
                        max_score_shape = (score, ((x + (0 + offset) * dx, y + (0 + offset) * dy),
                                                   (x + (1 + offset) * dx, y + (1 + offset) * dy),
                                                   (x + (2 + offset) * dx, y + (2 + offset) * dy),
                                                   (x + (3 + offset) * dx, y + (3 + offset) * dy),
                                                   (x + (4 + offset) * dx, y + (4 + offset) * dy)),
                                           (dx, dy))

        if max_score_shape[1] is not None:
            # 已统计的进攻棋形
            for score_piece in my_score_list:
                # 当前进攻棋形
                for pt1 in score_piece[1]:
                    for pt2 in max_score_shape[1]:
                        # 出现相同位置的不同方向出现进攻棋形
                        if pt1 == pt2 and max_score_shape[0] > 10 and score_piece[0] > 10:
                            # 分数增加 鼓励同个棋子凑多个棋形
                            add_score += score_piece[0] + max_score_shape[0]
            # 增加新进攻棋形
            my_score_list.append(max_score_shape)
        # 最终得分
        return add_score + max_score_shape[0]

    def _ai_get_pos(self):
        _, best_pos = self._alphabeta('AI', -999999999999, 9999999999999, 3)
        return best_pos

    def _has_neightnor(self, pos):
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if (pos[0] + i, pos[1] + j) in self.AI_pieces+self.player_pieces:
                    return True
        return False

    def _alphabeta(self, AI_or_Player, alpha, beta, depth):
        # 叶节点或达到搜索深度，直接返回
        if depth == 0 or self._win(self.AI_pieces) or self._win(self.player_pieces):
            return -self._eva() if AI_or_Player == 'AI' else self._eva(),(-1, -1)
        # 初始化最优落子点
        best_pos = (-1, -1)
        # 获得所有可落子的位置
        blank_pos_list = list(set(self.pos_in_chessboard).difference(set(self.AI_pieces+self.player_pieces)))
        for blank_pos in blank_pos_list:
            # 如果该空白位置周围没有棋子，则不作考虑
            if not self._has_neightnor(blank_pos):
                continue
            # 进入下一层, 负值最大算法
            if AI_or_Player == 'AI':
                self.AI_pieces.append(blank_pos)
                value, _ = self._alphabeta('Player', -beta, -alpha, depth - 1)
            else:
                self.player_pieces.append(blank_pos)
                value, _ = self._alphabeta('AI', -beta, -alpha, depth - 1)
            value = - value
            if AI_or_Player == 'AI':
                self.AI_pieces.remove(blank_pos)
            else:
                self.player_pieces.remove(blank_pos)
            if value > alpha:
                if depth == 3:
                    best_pos = blank_pos
                # alpha-beta剪枝
                if value >= beta:
                    return beta, best_pos
                alpha = value
        return alpha, best_pos

    def _play(self):
        begin = self.first
        while True:
            # 平局
            if len(list(set(self.pos_in_chessboard).difference(set(self.AI_pieces+self.player_pieces)))) == 0:
                break
            self._print_score()
            if begin == 'AI':
                pos = self._ai_get_pos()
                self.AI_pieces.append(pos)
                self._print_piece(pos,'white')
                if self._win(self.AI_pieces):
                    r = Rectangle(Point(0,0),Point(520,520))
                    r.setFill('red')
                    r.draw(self.chessboard)
                    Text(r.getCenter(),"你输了，点击以退出。").draw(self.chessboard)
                    break
                begin = 'Player'
            else:
                pos = self.chessboard.getMouse()
                pos = (round((pos.getX()-3*self.grid_width/2) / self.grid_width), round((pos.getY()-3*self.grid_width/2) / self.grid_width))
                if pos in list(set(self.pos_in_chessboard).difference(set(self.AI_pieces+self.player_pieces))):
                    self.player_pieces.append(pos)
                    self._print_piece(pos,'black')
                    if self._win(self.player_pieces):
                        r = Rectangle(Point(0, 0), Point(520, 520))
                        r.setFill('red')
                        r.draw(self.chessboard)
                        Text(r.getCenter(), "你赢了，点击以退出。").draw(self.chessboard)
                        break
                    begin = 'AI'
        self.chessboard.getMouse()
        return
if __name__ == '__main__':
    win = GraphWin("五子棋", 440, 440)
    win.setBackground('pink')
    r1= Rectangle(Point(160,200),Point(280,240)).draw(win)
    r2 = Rectangle(Point(160, 260), Point(280, 300)).draw(win)
    t1=Text(r1.getCenter(),'AI先手').draw(win)
    t2=Text(r2.getCenter(), '玩家先手').draw(win)
    t3=Text(Point(220, 110), '点击选择先后手开始五子棋').draw(win)
    while True:
        p = win.getMouse()
        if p.getX()>=160 and p.getX()<=280 and p.getY()<=240 and p.getY()>=200:
            first = 'AI'
            win.close()
            break
        if p.getX()>=160 and p.getX()<=280 and p.getY()<=300 and p.getY()>=260:
            first = 'Player'
            win.close()
            break
    play = GameTree(first)
