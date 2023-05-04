## 树



### 搜索

深度优先搜索利用函数递归或栈的先入后出结构实现，广度优先搜索利用队列的先入先出结构实现。

#### 1036. 逃离大迷宫

* bfs。网格过大，需要考虑提前退出的机制，进行<u>有限步数的搜索</u>。从障碍点个数入手，其包围的最佳方式是与网格的边够丑等腰直角三角形，可以得到包围圈中的最大点数。起点和终点进行双向bfs，检查其是否被围住。

```python
class Solution:
    def isEscapePossible(self, blocked: List[List[int]], source: List[int], target: List[int]) -> bool:
        def bfs(start,end):
            x1, y1 = start
            q = deque([(x1,y1)])
            visited = set([(x1,y1)])
            cnt = 1
            while q and cnt<=area:
                x, y = q.popleft()
                for i,j in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                    if [i,j] == end:
                        return 1
                    if 0<=i<Boundary and 0<=j<Boundary and (i,j) not in block and (i,j) not in visited:
                        cnt += 1
                        visited.add((i,j))
                        q.append((i,j))
            return 0 if cnt>area else -1

        n = len(blocked)
        Boundary = 10**6
        if n < 2:
            return True
        area = n*(n-1)//2
        block = set(tuple(each) for each in blocked)
        res = bfs(source,target)
        if res == 1:
            return True
        elif res == -1:
            return False
        else:
            res = bfs(target,source)
            if res != -1:
                return True
            return False
```



#### 1376.通知所有员工所用的时间

* dfs

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        graph = [[] for _ in range(n)]
        for i,x in enumerate(manager):  	# 邻接矩阵
            if x == -1:
                continue
            graph[x].append(i)
        res = 0
        cur = [(headID,informTime[headID])]	
        while cur:
            idx, cost = cur.pop()
            if not graph[idx]:
                res = max(res,cost)    	 	# 搜索到底
                continue
            for each in graph[idx]:
                cur.append((each,cost+informTime[each]))
        return res
```

```python
# 递归调用
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        graph = [[] for _ in range(n)]
        for i,x in enumerate(manager):  
            if x == -1:
                continue
            graph[x].append(i)
        
        def dfs(idx):
            if not graph[idx]:
                return 0
            cur = 0
            for each in graph[idx]:
                cur = max(cur,informTime[idx]+dfs(each))
            return cur

        return dfs(headID)
```

* bfs

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        graph = [[] for _ in range(n)]
        for i,x in enumerate(manager):  # 邻接矩阵
            if x == -1:
                continue
            graph[x].append(i)
        res = 0
        cur = deque([(headID,informTime[headID])])
        while cur:
            idx, cost = cur.popleft()
            if not graph[idx]:
                res = max(res,cost)
                continue
            for each in graph[idx]:
                cur.append((each,cost+informTime[each]))
        return res
```



