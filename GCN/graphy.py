import numpy as np

class Graph():

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]#对角线
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link#拼在一起
            self.center = 1
            #self.edge=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
             #(12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (4, 3), (3, 2), (7, 6), (6, 5), (13, 12),
             #(12, 11), (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]


    def get_adjacency(self, strategy):#'uniform'
        valid_hop = range(0, self.max_hop + 1, self.dilation)#range(0,2,1) [0,1]
        adjacency = np.zeros((self.num_node, self.num_node))#[18,18]
        for hop in valid_hop:#[0,1]
            # hop_dis=[[ 0.  1. inf inf inf inf inf inf inf inf inf inf inf inf  1.  1. inf inf]
            # [ 1.  0.  1. inf inf  1. inf inf inf inf inf inf inf inf inf inf inf inf]
            # [inf  1.  0.  1. inf inf inf inf  1. inf inf inf inf inf inf inf inf inf]
            # [inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf inf inf inf]
            # [inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf inf inf inf]
            # [inf  1. inf inf inf  0.  1. inf inf inf inf  1. inf inf inf inf inf inf]
            # [inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf]
            # [inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf]
            # [inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf inf inf inf]
            # [inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf]
            # [inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf]
            # [inf inf inf inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf]
            # [inf inf inf inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf]
            # [inf inf inf inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf]
            # [ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1. inf]
            # [ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1.]
            # [inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0. inf]
            # [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0.]]
            adjacency[self.hop_dis == hop] = 1
        #adjacency见150行

        normalize_adjacency = normalize_digraph(adjacency)#反正是进行了数学处理，不用知道干了什么

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            #用spatial
            A = []
            for hop in valid_hop:#[0,1]
                a_root = np.zeros((self.num_node, self.num_node))#[18,18]
                a_close = np.zeros((self.num_node, self.num_node))#[18,18]
                a_further = np.zeros((self.num_node, self.num_node))#[18,18]
                for i in range(self.num_node):#列[0,1,2,3,4...]
                    for j in range(self.num_node):#行[0,1,2,3,4...]
                        #hop_dis=
                        #[[ 0.  1. inf inf inf inf inf inf inf inf inf inf inf inf  1.  1. inf inf]
                        # [ 1.  0.  1. inf inf  1. inf inf inf inf inf inf inf inf inf inf inf inf]
                        # [inf  1.  0.  1. inf inf inf inf  1. inf inf inf inf inf inf inf inf inf]
                        # [inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf inf inf inf]
                        # [inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf inf inf inf]
                        # [inf  1. inf inf inf  0.  1. inf inf inf inf  1. inf inf inf inf inf inf]
                        # [inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf]
                        # [inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf]
                        # [inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf inf inf inf]
                        # [inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf]
                        # [inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf]
                        # [inf inf inf inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf]
                        # [inf inf inf inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf]
                        # [inf inf inf inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf]
                        # [ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1. inf]
                        # [ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1.]
                        # [inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0. inf]
                        # [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0.]]
                        if self.hop_dis[j, i] == hop:#hop先等于0,第二次等于1
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))#(18,18)
    for i, j in edge:
        # self.edge=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
        # (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (4, 3), (3, 2), (7, 6), (6, 5), (13, 12),
        # (12, 11), (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf#正无穷
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]#d=0,1,A的0和1次方
    arrive_mat = (np.stack(transfer_mat) > 0)#两个点之间值如果是1则代表它们相连，设置为true 三维[2,18,18]
    #当d=0只有对角线上是true，当d=1时transfer_mat[1]的值为1的地方都为true
    for d in range(max_hop, -1, -1):#d=1,0
        hop_dis[arrive_mat[d]] = d

    return hop_dis #其实是把各个点之间距离算出来了
    #hop_dis=[[ 0.  1. inf inf inf inf inf inf inf inf inf inf inf inf  1.  1. inf inf]
    #[ 1.  0.  1. inf inf  1. inf inf inf inf inf inf inf inf inf inf inf inf]
    #[inf  1.  0.  1. inf inf inf inf  1. inf inf inf inf inf inf inf inf inf]
    #[inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf inf inf inf]
    #[inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf inf inf inf]
    #[inf  1. inf inf inf  0.  1. inf inf inf inf  1. inf inf inf inf inf inf]
    #[inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf]
    #[inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf]
    #[inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf inf inf inf]
    #[inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf]
    #[inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf]
    #[inf inf inf inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf]
    #[inf inf inf inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf]
    #[inf inf inf inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf]
    #[ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1. inf]
    #[ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1.]
    #[inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0. inf]
    #[inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0.]]

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD



#[[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
# [1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0.]
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0.]
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0.]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1.]]