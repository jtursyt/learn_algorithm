## 图

### 相关概念

### 强连通图

在有向图中, 若对于每一对顶点v1和v2, 都存在一条从v1到v2和从v2到v1的路径,则称此图是强连通图。

#### 弱连通图

将有向图的所有的有向边替换为无向边，所得到的图称为原图的基图。如果一个有向图的基图是连通图，则有向图是弱连通图。

### 图的存储

#### 直接存边

使用数组存储各边的起点、终点和权值。

一般用于kruskal算法中对边权进行排序。

#### 邻接矩阵

使用一个二维数组来存边，m\[u][v]存储u到v的边的权值。

适用于没有重边的情况，适用于稠密图（边的数量接近点数量的平方），查询速度快。

#### 邻接表

使用一个支持动态增加元素的数据结构构成的数组，m[u]存储点u所有出边的相关信息。

适用于边数较少的稀疏图（边数约等于点数）。

使用链表实现的邻接表，称为链式前向星。

## 最短路

### 单源最短路



## 欧拉回路

### 定义

如果图G中的一个路径包括每个边恰好一次，则该路径称为**欧拉路径**。如果一个回路是欧拉路径，则称为欧拉回路。具有欧拉回路的图称为欧拉图，具有欧拉路径但不具有欧拉回路的图称为半欧拉图。

一个无向图存在欧拉回路，当且仅当该图所有**顶点度数都为偶数**，且该图是连通图。一个有向图存在欧拉回路，当且仅当所有**顶点的入度等于出度**，且该图是连通图。

一个图是半欧拉图的充分必要条件是其连通且仅有两个奇度点。

### Hierholzer算法

从某节点u开始（半欧拉图应为起点），任意的经过还未经过的边，直到无路可走，此时回到了该节点u（半欧拉图为到达终点v），得到了一条从u-->u的回路。接下来找到回路上具有出边的节点w，从w开始找到一条w-->w的回路嵌入原回路中，直到所有的边都被处理。通过递归操作，实际上是逆序确定回路中的边。

### 题目

#### 322. 重新安排行程

#### 753. 破解保险箱

* 转化为求欧拉回路。当前节点为a1a2...an-1，那么其连接的下一个节点为输入一个字符后的结果，如a2...an-1x。当前节点加上其出边的编号0~n即构成了一组密码，由于这个图的每个节点都有k条入边和出边，因此它一定存在一个欧拉回路，即可以从任意一个节点开始，一次性不重复地**走完所有的边**且回到该节点。

```python
class Solution:
    def crackSafe(self, n: int, k: int) -> str:
        def dfs(node):
            for x in range(k):
                nxt = node*10+x
                if nxt not in visited:
                    visited.add(nxt)
                    dfs(nxt%lims)
                    res.append(str(x))
        
        lims = 10**(n-1)
        visited = set()
        res = []
        dfs(0)
        return ''.join(res)+'0'*(n-1)
```

#### 2097. 合法重新排列数对



## 特殊结构

### 基环树

* 满足**恰好包含一个环**的无向连通图称作基环树；每个点入度都为1的有向弱连通图称作基环外向树；每个点出度都为1的有向弱连通图称作基环内向树。

* 解决方法——通过拓扑排序可以区分树枝和基环，排序完成后：
  * 从入度为1的点出发遍历基环
  * 根据反图搜索入度为0的点遍历树枝

#### 5970. 参加会议的最多员工数

* 由于大小为k的连通块具有k个点和k条边，所以一定存在一个环，根据环是否由两个节点构成可以分为两种情况。当基环节点数大于2时，最大基环的节点数即为圆桌最多员工数；基环等于2时，该基环所在的最长链中的节点数为最多相邻员工数，可将多个链进行拼接得到圆桌最多员工数。两种坐法进行比较得到最终解。

```python
class Solution:
    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        deg = [0] * n               # 入度
        for each in favorite:
            deg[each] += 1
        q = deque([i for i,each in enumerate(deg) if each==0])
        max_depth = [0] * n
        while q:                    # 拓扑排序
            cur = q.popleft()
            max_depth[cur] += 1     # 统计树枝上各节点所在链的最大长度
            nxt = favorite[cur]
            # 将链的长度传递到相连的环节点上
            max_depth[nxt] = max(max_depth[cur],max_depth[nxt]) 
            deg[nxt] -= 1
            if deg[nxt] == 0:
                q.append(nxt)
        max_ring = sum_chain = 0    # 分类讨论
        for i,d in enumerate(deg):
            if d!=0:
                deg[i] = 0
                ring_size = 1
                nxt = favorite[i]
                while nxt != i:     # 计算基环大小
                    deg[nxt] = 0
                    ring_size += 1
                    nxt = favorite[nxt]
                if ring_size==2:
                    sum_chain += max_depth[i]+max_depth[favorite[i]]+2
                else:
                    max_ring = max(max_ring, ring_size)
        return max(max_ring,sum_chain)
```

