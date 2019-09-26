# 论文名：Making Communities Show Respect for Order
from collections import defaultdict
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
path = 'data_test.txt'
lamda = 1  # 分辨系数
edges = []
nodes_suc = defaultdict(set)
nodes_pre = defaultdict(set)
with open(path, 'r') as f:
    for row in f:
        line = row.split()
        line[0] = int(line[0])
        line[1] = int(line[1])
        edges.append((line[0], line[1], 1))  # 得到网络中所有边的信息列表
        nodes_suc[line[0]].add(line[1])  # 得到拥有子节点的节点及其对应的子节点集合
        nodes_pre[line[1]].add(line[0])  # 得到拥有母节点的节点及其对应的母节点集合
nodes = nodes_pre.keys() | nodes_suc.keys()  # 网络的节点集合

# 以下就是算法主体了
Final_Coms_list = []  # 每轮压缩后的社团划分结果，对应的压缩关系也包含其中
num_out = 0
label_out = True
while label_out:

    # 用networkx形成G，等下在判断antichain时会用到
    G = nx.DiGraph()
    G.add_edges_from([edge[0:2] for edge in edges])  # 由于只是为了查看路径，这里的图没有考虑权重
    nx.draw_networkx(G, pos=nx.spring_layout(G))
    # plt.show()

    num_out = num_out+1
    # 第一阶段
    # 社团初始化
    Initial_Coms = []
    for x in nodes:
        Initial_Coms.append({x})  # 起初每个节点自成一个社团
    print('初始社团:', Initial_Coms)
    # 构建邻接矩阵和相似矩阵
    adj_matrix = np.array(nx.to_numpy_matrix(G))
    sim_matrix = np.dot(adj_matrix, adj_matrix.T)  # 用于siblinarty的计算

    num = 0  # 循环次数
    while 1:
        num = num+1  # 当循环大于一定次数后，自动中断
        label = False  # 每一轮节点遍历结束后，label恢复False。考虑大规模网络下的算法速度，遍历轮数被num的最大值限制。
        for node in nodes:
            Middle_Coms = Initial_Coms.copy()
            # 形成一个中间划分，排除当前节点所在社团
            for com in Initial_Coms:
                if node in com:
                    break
            com_begin = com.copy()   # 记录下当前节点所在的初始社团，在计算siblinarty的时候要用到
            com_select = com_begin.copy()
            delta_s_max = 0
            Middle_Coms.remove(com)  # 接下来就是当前节点分别并入各中间社团判断antichain和siblinarty了
            for com in Middle_Coms:
                s_after = 0
                s_before = 0
                # 第一，判断当前节点并入社团后是否依然形成antichain,只要确定当前节点和社团内各节点不存在路径即可
                path_label = False  # 节点间是否有路径的初始标签都是False
                for node_in_com in com:
                    if nx.has_path(G, node, node_in_com) | nx.has_path(G, node_in_com, node):
                        path_label = True
                        break
                # 第二，满足antichain后，判断siblinarty是否增加
                if not path_label:
                    # 先计算当前节点并入社团后，加入后这个社团的siblinarty，不能按照公式来
                    com1 = com.copy()
                    com1.add(node)   # com1表示当前节点并入某社团后得到的社团
                    for n in com1:
                        for m in com1:
                            if m > n:
                                kn = sum(sim_matrix[n-1])
                                km = sum(sim_matrix[m-1])
                                W = sum(sum(sim_matrix))
                                s_after = s_after+sim_matrix[m-1][n-1]-lamda*kn*km/W

                    # 再计算当前节点离开原社团后，原社团的siblinarty
                    com2 = com_begin.copy()
                    com2.remove(node)  # com2表示当前节点的初始社团去掉当前节点后得到的社团
                    if com2 != {}:
                        for n in com2:
                            for m in com2:
                                if m > n:
                                    kn = sum(sim_matrix[n-1])
                                    km = sum(sim_matrix[m-1])
                                    s_before = s_before+sim_matrix[m-1][n-1]-lamda*kn*km/W
                    # 计算siblinarty变化之差
                    delta_s = s_after - s_before
                    # 记录大于0的delta_s对应的社团
                    if delta_s > delta_s_max:
                        delta_s_max = delta_s
                        com_select = com.copy()
                        label = True  # 当社团发生变化时，label改变为True
            # 更新初始社团Initial_Coms
            if com_select != com_begin:
                Initial_Coms.remove(com_begin)
                Initial_Coms.append(com_begin-{node})
                Initial_Coms.remove(com_select)
                Initial_Coms.append(com_select|{node})
            if set() in Initial_Coms:
                Initial_Coms.remove(set())
        if (num >= 20) | (label == False):
            break
    print('第一阶段结束后的初始社团:', Initial_Coms)
    # 当最外面的循环执行到第二次及以上时，这时第一阶段得到的初始社团是压缩后的社团，但是在整个程序的循环中我们不需要将它们恢复过来，因为这是算法的要求
    # 但它是我们想要的结果，从这个角度看，我们需要把它给恢复过来，但又不能影响整个程序，所以我先把每次的划分结果都存在一个列表里面，同时也包含了对应关系
    # 我们只需要在程序运行结束后再将它恢复出来就好了
    Final_Coms_list.append(Initial_Coms)

    # 第二阶段：将第一阶段结束后得到的社团当作节点，以某种方式建立一个新的网络
    new_node_list = list(range(1, len(Initial_Coms)+1))  # 就按第一阶段结束后得到的社团的顺序重新编号，从1开始
    # 开始求新节点之间的边存在与否以及边的权重
    new_edges = []
    for i in range(len(Initial_Coms)):
        x = Initial_Coms[i]
        for j in range(len(Initial_Coms)):
            y = Initial_Coms[j]
            # 判断是否有边和边权重
            value = 0
            # 这里不防叫做x大节点和y大节点吧，x>y方向的边的权重由x内各节点的子节点和y内各节点的交集的节点数之和决定
            # 或者说是由y内个节点的母节点和x内各节点的交集的节点数之和决定，二者是等价的。
            if i != j:
                for node_in_x in x:
                    nodes_common = nodes_suc[node_in_x] & y  # x>y方向
                    # 计算权重
                    for node_in_nodes_commom in nodes_common:
                        if (node_in_x, node_in_nodes_commom) in [e[0:2] for e in edges]:
                            value = value + edges[[e[0:2] for e in edges].index((node_in_x, node_in_nodes_commom))][2]
            new_edges.append((new_node_list[i], new_node_list[j], value))
    # 虽说以上操作可以得到新的网络图了，但是边中有很多权重为0的，即等于没有边，为了好看，我们将它们删掉
    for edge in new_edges.copy():
        if edge[2] == 0:
            new_edges.remove(edge)
    print('new_edges:', new_edges)
    # 得到新的网络图后，我们回头看第一阶段和整个阶段，其实只需要网络的三个参数: edges, nodes, nodes_suc
    # 分别对应于new_edges, new_nodes, new_nodes_suc
    new_nodes = set(new_node_list)
    new_nodes_suc = defaultdict(set)
    for e in new_edges:
        new_nodes_suc[e[0]].add(e[1])

    # 2个阶段都结束了，替换前面提到的三个参数
    edges = new_edges.copy()
    nodes = new_nodes.copy()
    nodes_suc = new_nodes_suc.copy()

    # 包含2个阶段的外循环大于等于30次时或者当第二阶段不再形成新的网络时中断
    if (len(Final_Coms_list)) > 1:
        if (Final_Coms_list[-1] == Final_Coms_list[-2]) | (num_out >= 30):
            label_out = False
# print('Final_Coms_list（有重复）:', Final_Coms_list)
# 开始在Final_Coms_list中恢复出最终社团划分Final_Coms了
Final_Coms_list.remove(Final_Coms_list[-1])  # 最后一个重复了，就删除
print('Final_Coms_list:', Final_Coms_list)
Final_Coms_list.reverse()
Final_Coms = Final_Coms_list[0]
for i in range(len(Final_Coms_list)-1):  # 这个i是对Final_Coms_list各项的索引
    dict = {}
    for j in range(len(Final_Coms_list[i+1])):
        dict[j+1] = Final_Coms_list[i+1][j]  # 对应关系就找到了：前一项社团划分中的节点数字的含义，是指向一个节点集
    # 既然对应关系找到了，那么前一项社团就可以用后一项社团表示了
    s = 0
    for cluster in Final_Coms.copy():
        replace_cluster = set()
        for n in cluster:
            replace_cluster.update(dict[n])
        Final_Coms[s] = replace_cluster
        s = s+1
print('最终划分结果:', Final_Coms)






