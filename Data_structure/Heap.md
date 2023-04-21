## 堆

二进制堆是一种完全二叉树形式的数据结构（数组存储二叉树层序遍历结果），同时满足堆的性质：父节点值大于（大顶堆）或小于子节点值（小于）。由于采用数组的形式存储，内存的占用比普通的树小，用堆的目的是将最大（或者最小）的节点放在最前面，从而快速的进行相关插入、删除操作。

一般用于构建优先队列，进行堆排序，快速找出集合中的最大（最小）值。优先队列可以在O(1)时间内获得最值，并且可以在O(logn)时间内取出最值或插入人一致。

### 构建堆

```python
# 大顶堆
class MaxHeap:
    def __init__(self):
        self.heap = []
        self.cnt = 0
        
    # 将新节点插入堆最底部，整理堆结构
    def push(self, x):
        self.heap.append(x)
        self.shift_up(self.cnt)
        self.cnt += 1
    
    # 将堆顶节点输出，将堆底节点放置到堆顶，整理堆结构
    def pop(self):
        if self.cnt:
            ans = self.heap[0]
            self.cnt -= 1
            self.heap[0] = self.heap[-1]
            self.heap.pop()
            self.shift_down(0)
            return ans
        
	# 将当前节点与父节点交换位置
    def shift_up(self,idx):
        parent = (idx-1) >> 1
        while idx>0 and self.heap[parent]<self.heap[idx]:
            self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]
            idx = parent
            parent = (idx-1) >> 1
    
    # 将当前节点与较大的子节点交换位置
    def shift_down(self,idx):
        lc = (idx<<1) + 1	# 左节点
        while lc < self.cnt:
            if lc+1 < self.cnt and self.heap[lc+1] > self.heap[lc]:
                lc += 1		# 选择更大的右节点
            if self.heap[lc] > self.heap[idx]:
                self.heap[lc], self.heap[idx] = self.heap[idx], self.heap[lc]
                idx = lc
                lc = (idx<<1) + 1
            else:
                break
```

### heapq

* python内置的小顶堆模块

```python
heapq.heapify(heap)    # 将列表转化为堆结构
heapq.heappush(heap,num)   # 将数值加入堆
heapq.heappop(heap)        # 弹出最小值
heapq.heappushpop(heap, num)	   # 加入num后弹出最小值
heapq.heapreplace(heap,num)     # 替换最小值
heapq.nlargest(k,heap,[key=])     # 堆中最大的k个元素
heapq.nsmallest(k,heap,[key=])	   # 堆中最小的k个元素
heapq.merge(*iterables, key=None, reverse=False)	# 将多个已排序的输入合并为一个已排序的输出，返回已排序值的 iterator。
```

### [堆排序](../排序/排序)

### 题目

#### 218. 天际线问题

* 大顶堆。遍历左右端点，遇到左端点时大顶堆记录建筑的高度及其右端点，遇到右端点时将所有小于其的建筑出队，当前的最高建筑出现变化时，保存该节点。

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        points = []
        for l,r,h in buildings:
            # 利用负号区分左右端点
            points.append((l,-h,r))  
            points.append((r,h,0))
        points.sort()   # 从左到右扫描各端点
        pre = 0 
        heap, res = [(0,float('inf'))], []
        for p,h,r in points:
            if h < 0:
                heapq.heappush(heap,(h,r))
            else:       # 利用右端点延迟删除无关建筑
                 while heap[0][1] <= p:
                     heapq.heappop(heap)
            cur = -heap[0][0]      # 目前最高
            if cur != pre:
                res.append([p,cur])
                pre = cur
        return res
```

#### 253. 会议室 II

* 小顶堆，存储会议的结束时间，其容量代表需要同时进行的会议数。

```python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        heap = []
        for each in intervals:
            if heap and heap[0] <= each[0]:
                heapq.heappop(heap)
            heapq.heappush(heap,each[1])
        return len(heap)
```

#### 295. 数据流的中位数

* 双堆，大顶堆存储较小的一半，小顶堆存储较大的一半，维持二者数据量之差不超过1.

```python
class MedianFinder:
    def __init__(self):
        self.small = []
        self.big = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small,-num)
        heapq.heappush(self.big,-heapq.heappop(self.small))
        if len(self.big) > len(self.small)+1:
            heapq.heappush(self.small,-heapq.heappop(self.big))

    def findMedian(self) -> float:
        return (self.big[0]-self.small[0])/2  if len(self.big) == len(self.small) else self.big[0]
```

#### 347. 前k个高频元素

* 小顶堆，维护一个大小为k的小顶堆，记录元素频率。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        def shift_up(i):
            parent = (i-1) >> 1
            while i > 0 and cnt[heap[parent]] > cnt[heap[i]]:
                heap[parent], heap[i] = heap[i], heap[parent]
                i = parent
                parent = (i-1) >> 1
        
        def shift_down(i):
            l = (i<<1)+1
            while l < k:
                if l+1 < k and cnt[heap[l+1]] < cnt[heap[l]]:
                    l += 1
                if cnt[heap[l]] < cnt[heap[i]]:
                    heap[l], heap[i] = heap[i], heap[l]
                    i = l
                    l = (i<<1)+1
                else:
                    break
                    
        heap = []
        cnt = Counter(nums)
        num = list(cnt.keys())
        for i in range(k):
            heap.append(num[i])
            shift_up(i)
        for i in range(k,len(num)):
            if cnt[num[i]] > cnt[heap[0]]:
                heap[0] = num[i]
                shift_down(0)
        return heap
```

```python
# heapq实现
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        class Num:
            def __init__(self,num):
                self.num = num        
            def __lt__(self,other):
                return cnt[self.num] < cnt[other.num]

        heap = []
        cnt = Counter(nums)
        num = list(cnt.keys())
        for i in range(len(num)):
            if len(heap)<k:
                heapq.heappush(heap,Num(num[i]))
            elif cnt[num[i]] > cnt[heap[0].num]:
                heapq.heappushpop(heap,Num(num[i]))
        return [each.num for each in heap]
```

#### <span id="jump1"> 373. 查找和最小的K对数字</span>

* 大顶堆O(k^2logk)。维护一个大小为k的大顶堆。

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = []
        for a in nums1:
            for b in nums2:
                if len(heap) < k:
                    heapq.heappush(heap, (-(a + b), a, b))
                elif -(a + b) > heap[0][0]:
                    heapq.heappushpop(heap, (-(a + b), a, b))
                else:
                    break
        return [[i,j] for _,i,j in heap]
```

* 小顶堆(klogk)。利用两个数组升序的特点，(i, j)之后的下一个元素为(i+1,j)或(i,j+1)，逐步将$(i_0,i_1,..,i_n)+j_0,(i_0,i_1,...,i_n)+j_1...$各项中的元素入堆归并。

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        res = []
        heap = [(nums1[i]+nums2[0],i,0) for i in range(min(k,len(nums1)))]
        while heap and len(res)<k:
            _, i, j = heapq.heappop(heap)
            res.append([nums1[i],nums2[j]])
            if j+1<len(nums2):
                heapq.heappush(heap,(nums1[i]+nums2[j+1],i,j+1))
        return res
```

#### 407. 接雨水II

* 最小堆。从外围一周开始，找到最低点，最低点周围的水量由最低点的高度决定。

```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        m, n = len(heightMap), len(heightMap[0])
        if m<=2 or n<=2:
            return 0
        visited = [[0]*n for _ in range(m)]
        heap = []
        for i in range(m):
            for j in range(n):
                if i==0 or i==m-1 or j==0 or j==n-1:
                    visited[i][j] = 1
                    heapq.heappush(heap,(heightMap[i][j],i,j))
        res = 0
        dirs = [-1,0,1,0,-1]
        while heap:
            h, x, y = heapq.heappop(heap)
            for k in range(4):
                nx, ny = x+dirs[k], y+dirs[k+1]
                if 0<nx<m and 0<ny<n and not visited[nx][ny]:
                    visited[nx][ny] = 1
                    res += max(h-heightMap[nx][ny],0)
                    # 新节点的高度是二者中的最大值
                    heapq.heappush(heap,(max(h,heightMap[nx][ny]),nx,ny))
        return res
```

#### 480. 滑动窗口的中位数

* 双堆+延迟删除。进行 插入-->平衡-->删除-->平衡 的操作，窗口滑动时对于非堆顶的元素，哈希表记录应该删除的数量，待其位于堆顶时进行删除；同时记录堆中有效数字的数量，根据其进行平衡操作。两种情况下需要进行删除操作：
  * 直接删除：堆顶元素刚好为当前窗口最左边的元素；
  * 延迟删除：平衡或直接删除后产生新的堆顶元素是待删除元素。

```python
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        def insert(num):
            nonlocal len1, len2
            if not small or num <= -small[0]:
                heapq.heappush(small,-num)
                len1 += 1
            else:
                heapq.heappush(big,num)
                len2 += 1
            balance()
        
        # 平衡堆大小，找到中位数，并保证堆顶元素不是待删除元素
        def balance():
            nonlocal len1, len2
            if len1 > len2+1:
                heapq.heappush(big,-heapq.heappop(small))
                len1 -= 1
                len2 += 1
                check(small,-1)		  # 延迟删除
            elif len1 < len2:
                heapq.heappush(small,-heapq.heappop(big))
                len1 += 1
                len2 -= 1
                check(big,1)
        
        def check(heap,sign):
            nonlocal len1, len2
            while heap and d[sign*heap[0]]:
                d[sign*heap[0]] -= 1
                heapq.heappop(heap)
        
        def erase(i):
            nonlocal len1, len2
            d[nums[i]] += 1		      # 统计待删除元素数量
            if nums[i] <= -small[0]:  # 确定删除位置
                len1 -= 1
                check(small,-1)		  # 进行直接删除和延迟删除
            else:
                len2 -= 1
                check(big,1)
            balance()		# 平衡堆大小，保证small==big or small==big+1
        
        res = []
        d = defaultdict(int)    # 记录待删除元素
        n = len(nums)
        len1 = len2 = 0         # 记录两个堆中有效数字的数量
        small, big = [], []
        for i in range(k):
            insert(nums[i])
        res.append(-small[0] if len1>len2 else (big[0]-small[0])/2)
        for i in range(k,n):
            insert(nums[i])		# 插入，平衡
            erase(i-k)			# 删除，平衡
            res.append(-small[0] if len1>len2 else (big[0]-small[0])/2)
        return res
```

#### 692. 前k个高频单词

* 维护一个大小为k的小顶堆，重写\__lt__解决字母顺序排列问题。

```python
class Word:
    def __init__(self,word,freq):
        self.word = word
        self.freq = freq
    
    def __lt__(self,other):
        if self.freq != other.freq:
            return self.freq < other.freq
        return self.word > other.word

class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        count = Counter(words)
        heap = []
        for w,v in count.items():
            if len(heap) < k:
                heapq.heappush(heap,Word(w,v))
            elif v > heap[0].freq or (v == heap[0].freq and w<heap[0].word):
                heapq.heappushpop(heap,Word(w,v))
        res = []
        while True:
            try:
                res.append(heapq.heappop(heap).word)
            except:
                return res[::-1]
```

```python
# 手写小顶堆
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        def compare(a,b):
            return word[a] > word[b] or (word[a] == word[b] and a < b)

        def heappush(x):
            if self.cnt < k:
                res.append(x)
                shiftup(self.cnt)
                self.cnt += 1
            elif compare(x,res[0]):
                res[0] = x
                shiftdown(0)

        def heappop():
            if self.cnt:
                cur = res[0]
                self.cnt -= 1
                res[0] = res[-1]
                res.pop()
                shiftdown(0)
                return cur  
        
        def shiftup(idx):
            parent = (idx-1) >> 1
            while idx > 0 and compare(res[parent],res[idx]):
                res[parent],res[idx] = res[idx], res[parent]
                idx = parent
                parent = (idx-1) >> 1
        
        def shiftdown(idx):
            left = (idx<<1) + 1
            while left < self.cnt:
                if left+1 < self.cnt and compare(res[left],res[left+1]):
                    left += 1
                if compare(res[idx],res[left]):
                    res[left], res[idx] = res[idx], res[left]
                    idx = left
                    left = (idx<<1)+1
                else:
                    break
        res = []
        self.cnt = 0
        word = Counter(words)
        for each in word.keys():
            heappush(each)
        new = []
        while self.cnt:
            new.append(heappop())    
        return new[::-1]
```

#### 706. 数据流中的第k大元素

* 维护一个大小为k的小顶堆。

```python
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.hq = nums
        heapq.heapify(self.hq)
        while len(self.hq) > k:
            heapq.heappop(self.hq)

    def add(self, val: int) -> int:
        if len(self.hq) < self.k:
            heapq.heappush(self.hq, val)
        elif val > self.hq[0]:
            heapq.heapreplace(self.hq, val)
        return self.hq[0]
```

#### 846. 一手顺子

* 每次取出最小的牌，检查是否存在连牌。

```python
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize:
            return False
        d = Counter(hand)
        heapq.heapify(hand)
        while hand:
            cur = heapq.heappop(hand)
            if d[cur]:
                for i in range(cur,cur+groupSize):
                    d[i] -= 1
                    if d[i] < 0:
                        return False
        return True
```

#### 1046. 最后一块石头的重量

* 大顶堆

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = list(map(lambda x:-x,stones))
        heapq.heapify(heap)
        while heap:
            s1 = heapq.heappop(heap)
            if heap:
                s2 = heapq.heappop(heap)
            else:
                return -s1
            heapq.heappush(heap,-abs(s1-s2))
        return 0
```

#### 1488.避免洪水泛滥

* 小顶堆，获得湖泊抽水的顺序。

```python
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        lake = defaultdict(deque)   
        for i,each in enumerate(rains):     # 统计每个湖的下雨日期
            if each:
                lake[each].append(i)
        res = [1]*len(rains)
        q = []
        full = set()
        for i,each in enumerate(rains):
            if each != 0:
                res[i] = -1
                if each in full:   		# 发洪水
                    return []
                else:
                    full.add(each)		
                    lake[each].popleft()
                    if lake[each]:
                        heapq.heappush(q,lake[each][0])   # 将抽水的deadline入堆
            elif q:
                idx = heapq.heappop(q)      # 抽取最近会下雨的湖泊
                res[i] = rains[idx]
                full.remove(rains[idx])
        return res
```

#### 1705. 吃苹果的最大数目

* 小顶堆，优先队列确定最早坏的苹果先吃。

```python
class Solution:
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        n = len(apples)
        heap = []
        res = 0
        for i in range(n):
            while heap and heap[0][0] <= i:
                heapq.heappop(heap)
            if apples[i]:
                heapq.heappush(heap,[days[i]+i,apples[i]])	# 入堆保质期和存量
            if heap:
                heap[0][1] -= 1
                if heap[0][1]==0:
                    heapq.heappop(heap)
                res += 1
        i += 1
        while heap:
            while heap and heap[0][0] <= i:
                heapq.heappop(heap)
            if heap:
                nxt = min(heap[0][0]-i,heap[0][1])
                heapq.heappop(heap)
                res += nxt
                i += nxt
        return res
```



