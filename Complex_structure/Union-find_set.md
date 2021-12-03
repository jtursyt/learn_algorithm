## 并查集

#### 2092. 找出知晓秘密的所有专家

* 对会议按照时间排序，对每个时间段的会议遍历两次，第一次将两个专家连接，若其中有人知晓秘密则都连接到专家0，第二次检查已连接的两个专家是否能知晓秘密，如果不能删除连接，反之连接到专家0。

```python
class Unionfind:
    def __init__(self,n):
        self.parent = list(range(n))
        self.rank = [1]*n
    
    def find(self,x):
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if x and self.rank[x] < self.rank[y]:   # 一定将专家0作为祖先
            x, y = y, x
        self.parent[y] = x
        self.rank[x] += self.rank[y]
    
    def delete(self,x):
        self.parent[x] = x
        self.rank[x] = 1

class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        d = defaultdict(list)
        uf = Unionfind(n)
        uf.union(0,firstPerson)
        for x,y,t in meetings:
            d[t].append((x,y))
        for t in sorted(d.keys()):
            for x,y in d[t]:
                if not (uf.find(x) and uf.find(y)):
                    uf.union(0,x)
                    uf.union(0,y)
                else:
                    uf.union(x,y)
            for x,y in d[t]:
                if not (uf.find(x) and uf.find(y)):
                    uf.union(0,x)
                    uf.union(0,y)
                else:
                    uf.delete(x)
                    uf.delete(y)
        return [i for i in range(n) if not uf.find(i)]
```

