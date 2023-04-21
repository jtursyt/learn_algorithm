## 动态规划

DP算法把原问题视作若干个**重叠子问题**的逐层递进，每个子问题的求解过程都构成一个<u>阶段</u>。在完成前一个阶段的计算后，动态规划才会执行下一阶段的计算。

> 重叠子问题：子问题与原问题的计算步骤一样，且在计算原问题时需要多次重复计算子问题。

为了保证计算能够按顺序、不重复地进行，动态规划要求已经求解的子问题不受后续阶段的影响，即**无后效性**（动态规划对<u>状态</u>空间的遍历构成一张有向无环图，遍历顺序就是该有向无环图的一个拓扑序，而边的选择就是动态规划中的<u>决策</u>）。

> *自底向上* 的动态规划算法是按 *逆拓扑序* 来处理子问题图中的顶点。对于任何子问题，直至它依赖的所有子问题均已求解完成才会求解它。因此这种方法需要恰当定义子问题的“规模”，使得任何子问题的求解都只依赖于更小的子问题的求解。
>
> *带备忘的自顶向下* 的动态规划算法相当于在子问题图中进行深度优先搜索。此方法仍按自然的递归形式编写过程，但过程中会保存每个子问题的解。
>
> 通常情况下，如果每个子问题都必须求解，自底向上的方法会更快，因为其没有递归调用的开销，表的维护开销也更小。但若一些子问题没有求解的必要时，自顶向下的方法更占优势，因为它只会求解绝对必要的子问题，

动态规划在阶段计算完成时，只会在每个状态上保留与最终解集相关的部分代表信息，代表信息应具有 *可重复的求解过程* ，并能够导出后续阶段的代表信息。体现在优化问题中即是**最优子结构性质**。

> 最优子结构: 动态规划常用于求解最优化问题，问题的最优解由相关子问题的最优解组合而成，而这些子问题可以独立求解。


动态规划算法把相同的计算过程作用于各阶段的同类子问题，这个计算过程即为 *状态转移方程* 。

解决步骤：

* 刻画一个最优解的结构特征；
* 递归地定义最优解的值；
* 计算最优解的值；
* 利用计算出的信息构造一个最优解。



多阶段决策最优解：在解决问题的过程中，需要经过多个决策<u>阶段</u>，每个决策阶段对应一组<u>状态</u>，寻找能够产生最优值的<u>决策</u>序列。

> 同样是解决多阶段决策最优解问题，回溯算法相当于穷举搜索，复杂度在指数级别；动态规划比回溯算法高效，但必须满足最优子结构、无后效性和重叠子问题这三个特征才能使用；贪心算法相当于动态规划的一种特殊情况，效率更高，在要求满足最优子结构和无后效性之外，需要满足贪心选择性——通过局部最优的选择能产生全局最优的结果。

> 分治法与动态规划相比，是将原问题划分为互不相交的子问题，递归的求解子问题，再将它们的解组合起来求原问题的解。适合用分治方法求解的问题通常在递归的每一步都生成全新的子问题，而不是求解重叠子问题。


### 线性DP

DP阶段沿各个维度线性增长，从一个或多个“边界点”开始有方向地向整个状态空间转移、扩展，最后每个状态上都保留了以自身为”目标“的子问题的最优解。

这类问题中，需要计算的对象表现出明显的维度以及有序性，每个状态的求解直接构成一个阶段，使得**DP的状态表示就是阶段的表示**。在每个维度上各取一个坐标值作为DP的状态，自然就可以描绘出“已求解部分”在状态空间中的轮廓特征。按顺序依次循环每个维度，根据问题要求递推求解即可。

#### 前缀和

##### 53. 最大子数组和

* dp[i]表示以i结尾的最大子数组和。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        for i in range(n):
            dp[i] = max(nums[i],nums[i]+dp[i-1])
        return max(dp)
```

* 前缀和

```python
class Solution(object):
    def maxSubArray(self, nums):
        sums = mins = 0
        res = float('-inf')
        for each in nums:
            sums += each
            res = max(res, sums-mins)
            mins = min(mins, sums)
        return res
```

* 贪心，当前子序和为负数时，重新计算。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = nums[0]
        cur = 0
        for each in nums:
            cur = each if cur <= 0 else cur+each
            res = max(res,cur)
        return res
```

##### 363. 矩形区域不超过k的最大数值和

* 前缀和+二分查找。 插入操作是O(n)，可以优化。

```python
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        m, n, res = len(matrix), len(matrix[0]), float('-inf')
        for l in range(n):
            sums = [0] * m
            for r in range(l,n):    
# 选择第l、r列作为矩阵的左右边界，上下边界是0和m-1，由于行数远大于列数所以用列作边界。 
                for i in range(m):  
                    sums[i] += matrix[i][r]   # 计算矩形每行的和
                presum = [0]
                cur = 0
                for row in sums:
                    cur += row     # 计算前缀和
                    # 二分查找与cur相减后等于k的前缀和
                    pos = bisect.bisect_left(presum,cur-k)  
                    if pos < len(presum):
                        res = max(res,cur-presum[pos])
                    bisect.insort(presum,cur)	# 保持前缀和有序
        return res
```

##### 303. 区域和检索-数组不可变

##### 304. 二维区域和检索-矩阵不可变

##### I17.24. 最大子矩阵

* 前缀和+最大子数组。遍历不同的行数作为矩阵的两条边，转换为一维的最大子数组，贪心求解。

```python
class Solution:
    def getMaxMatrix(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        prefix = [[0]*n for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                prefix[i+1][j] += prefix[i][j]+matrix[i][j]
        r1 = c1 = r2 = c2 = 0
        res = float('-inf')
        for i in range(m):
            for j in range(i,m):
                cur = [b-a for a,b in zip(prefix[i],prefix[j+1])]
                l = tmp = 0
                for r,each in enumerate(cur):
                    if tmp > 0:
                        tmp += each
                    else:
                        tmp = each
                        l = r
                    if tmp > res:
                        res = tmp
                        r1, c1, r2, c2 = i,l,j,r
        return r1,c1,r2,c2
```

##### 1477. 找两个和为目标值且不重叠的子数组

* dp[i]记录arr[:i+1]中和为target的最短子数组长度，字典记录前缀和帮助查找满足条件的子数组。

```python
class Solution:
    def minSumOfLengths(self, arr: List[int], target: int) -> int:
        n = len(arr)
        dp = [float('inf')] * n
        prefix = {0:-1}
        cur = 0
        res = float('inf')
        for i in range(n):
            cur += arr[i]
            dp[i] = dp[i-1]
            if cur-target in prefix:
                idx = prefix[cur-target]
                dp[i] = min(dp[i],i-idx)
                # 找到当前子数组前的最短子数组
                if idx >= 0 and dp[idx] != float('inf'):
                    res = min(res,dp[idx]+i-idx)
            prefix[cur] = i
        return res if res != float('inf') else -1
```

#### 特殊子序列

##### 70. 爬楼梯

* 到第n-1个台阶的走法 + 第n-2个台阶的走法 = 到第n个台阶的走法。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a, b = 1, 0
        for _ in range(n):
            a, b = a+b, a
        return a
```

##### 746. 使用最小花费爬楼梯

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        a, b = cost[1], cost[0]
        for i in range(2,len(cost)):
            a, b = min(a,b)+cost[i], a
        return min(a,b)
```

##### 509. 斐波那契数

* 矩阵的快速幂。

```python
class Solution:
    def m_pow(self,n,a):
        m = [[1,0],[0,1]]
        while n > 0:
            if n&1:
                m = self.multiply(m,a)
            n >>= 1
            a = self.multiply(a,a)
        return m

    def multiply(self,a,b):
        res = [[0,0],[0,0]]
        for i in range(2):
            for j in range(2):
                res[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j]
        return res

    def fib(self, n: int) -> int:
        if n<2:
            return n
        m = [[1,1],[1,0]]
        res = self.m_pow(n-1,m)
        return res[0][0]
```

##### 264. 丑数2

##### 413. 等差数列划分

##### 198. 打家劫舍

* dp[i]表示偷窃前i间房屋的最高金额。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0]*n
        dp[0], dp[1] = nums[0], max(nums[0],nums[1])
        for i in range(2,n):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        return dp[-1]
# 空间优化
class Solution:
    def rob(self, nums: List[int]) -> int:
        pre = cur = 0
        for each in nums:
            cur, pre = max(pre+each,cur), cur
        return cur
```

##### 213. 打家劫舍 II

* 按照是否偷第一家统计两种情况。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return max(nums)
        a = b = d = 0
        c = nums[0]
        for each in nums[1:]:
            a, b = max(a,b+each), a
            c, d = max(c,d+each), c
        return max(a,d)
```

##### 1911. 最大子序列交替和

* dp\[i][0]和dp\[i][1]分别记录前i个数中长为偶数和奇数的最大子序列交替和。

```python
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0]*2 for _ in range(n+1)]
        for i in range(1,n+1):
            dp[i][0] = max(dp[i-1][1]+nums[i-1],dp[i-1][0])
            dp[i][1] = max(dp[i-1][0]-nums[i-1],dp[i-1][1])
        return max(dp[-1])
```

* 贪心

```python
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        res = pre = 0
        for each in nums:
            if each > pre:
                res += each - pre
            pre = each
        return res
```

##### 740. 删除与获得点数

#### 最优解问题

##### 279. 完全平方数

##### 646. 最长数对链

##### 650. 只有两个键的键盘

##### 801. 使序列递增的最小交换次数

##### 813. 最大平均值和的分组

##### 1262. 可被三整除的最大和

##### 1537. 最大得分

##### 368. 最大整除子集	

* 先对数组排序，dp[i]记录以nums[i]结尾的最大整除子集。

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp = [[each] for each in nums]
        for i in range(1,n):
            for j in range(i-1,-1,-1):
                if not nums[i]%nums[j]:
                    dp[i] = max(dp[i],dp[j]+[nums[i]],key=len)
        return max(dp,key=len)
```

##### 256. 粉刷房子

* costs\[i][j]记录把房子i刷成j色的最小花费。

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        for i in range(1,len(costs)):
            costs[i][0] += min(costs[i-1][1],costs[i-1][2])
            costs[i][1] += min(costs[i-1][0],costs[i-1][2])
            costs[i][2] += min(costs[i-1][0],costs[i-1][1])
        return min(costs[-1])
```

##### 265. 粉刷房子2

##### 1473. 粉刷房子3

##### 121. 买卖股票的最佳时机

* dp[i]表示第i天卖出股票获得的最大收入。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * n
        for i in range(1,n):
            dp[i] = max(0,prices[i]+dp[i-1]-prices[i-1])	# 收入大于等于0
        return max(dp)
```

* 贪心，找到左侧的最小值和右侧的最大值。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_val = float('inf')
        res = 0
        for each in prices:
            if each > min_val:
                res = max(res,each-min_val)
            else:
                min_val = each
        return res
```

##### 122. 买卖股票的最佳时机 II

* dp\[i][1]表示第i天交易结束后持有股票时的最大利润，dp\[i][0]表示第i天卖出股票的最大利润。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0]*2 for _ in range(n)]
        dp[0][1] = -prices[0]
        for i in range(1,n):
            dp[i][0] = max(dp[i-1][1]+prices[i],dp[i-1][0])
            dp[i][1] = max(dp[i-1][0]-prices[i],dp[i-1][1])
        return dp[-1][0]
    
# 空间优化
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hold, sell = -prices[0],0
        for i in range(1,len(prices)):
            sell, hold = max(hold+prices[i],sell), max(sell-prices[i],hold)
        return sell
```

* 贪心，累加相邻项之间的收益。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0
        for i in range(1,len(prices)):
            diff = prices[i]-prices[i-1]
            if diff>0:
                res += diff
        return res
```

##### 123. 买卖股票的最佳时机 III

* dp\[i]\[j][k]三个维度分别是天数、交易次数、是否持有股票，记录当前的最大收益。第几次买入股票即为第几次交易。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[[0]*2 for _ in range(3)] for _ in range(n)]
        dp[0][1][1] = dp[0][2][1] = -prices[0]
        for i in range(1,n):
            for j in (1,2):
                dp[i][j][0] = max(dp[i-1][j][1]+prices[i],dp[i-1][j][0])
                dp[i][j][1] = max(dp[i-1][j-1][0]-prices[i],dp[i-1][j][1])
        return max(each[0] for each in dp[-1])
```

* 贪心，前后两次遍历，找到两次交易总利润最大的方式。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        profit = [0]*n
        min_val = prices[0]
        # 找到进行一次交易，第i天可以获得的最大利益。
        for i in range(1,n):
            min_val = min(min_val,prices[i])
            profit[i] = max(profit[i-1],prices[i]-min_val)	
        max_val = prices[-1]
        res = profit[-1]
        for i in range(n-1,1,-1):
            max_val = max(max_val,prices[i])
            res = max(res,max_val-prices[i]+profit[i-1])
        return res
```

##### 188. 买卖股票的最佳时机 IV

* 同上题。当k较大时，简化为122。

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        def greedy(prices):
            res = 0
            for i in range(1,n):
                if prices[i] > prices[i-1]:
                    res += prices[i]-prices[i-1]
            return res
        
        n = len(prices)
        if n < 2:
            return 0
        if k > n//2:
            return greedy(prices)
        dp = [[[0]*2 for _ in range(k+1)] for _ in range(n)]
        for j in range(1,k+1):
            dp[0][j][1] = -prices[0]
        for i in range(1,n):
            for j in range(1,k+1):
                dp[i][j][0] = max(dp[i-1][j][0],prices[i]+dp[i-1][j][1])
                dp[i][j][1] = max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i])
        return max(each[0] for each in dp[-1])

# 空间优化
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:   
        n = len(prices)
        if n < 2:
            return 0
        hold = [-prices[0]]*(k+1)
        sell = [0]*(k+1)
        for i in range(1,n):
            for j in range(k,0,-1):
                sell[j] = max(sell[j],hold[j]+prices[i])
                hold[j] = max(hold[j],sell[j-1]-prices[i])     
        return max(sell)
```

##### 309. 最佳买卖股票时机含冷冻期

* hold, froz, sell分别对应第i天持有股票，冷冻期和不持有股票。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sell = froz = 0
        hold = -prices[0]
        n = len(prices)
        for i in range(1,n):
            sell, hold ,froz = max(sell,froz), max(hold,sell-prices[i]), hold+prices[i]
        return max(sell,froz)
```

##### 714. 买卖股票的最佳时机含手续费

* 同122。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        hold, sell = -prices[0]-fee, 0
        for each in prices[1:]:
            hold, sell = max(hold,sell-each-fee), max(sell,hold+each)
        return sell
```

##### 983. 最低票价

* dp[i]表示1～i天内旅行的最小花费。

```python
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        dp = [0]*(days[-1]+1)
        for i in range(1,days[-1]+1):
            if i in days:
                dp[i] = min(dp[i-1]+costs[0],dp[max(i-7,0)]+costs[1],dp[max(i-30,0)]+costs[2])
            else:
                dp[i] = dp[i-1]
        return dp[-1]
```

##### 1105. 填充书架

* dp[i]表示放置book[i]需要的书架最小高度。

```python
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        n = len(books)
        dp = [0]+[float('inf')]*n
        for i in range(1,n+1):
            w, h = books[i-1]
            dp[i] = dp[i-1]+h
            # 判断第i本书能否和之前的书放到一层
            for j in range(i-1,0,-1):
                w += books[j-1][0]
                if w > shelfWidth:
                    break
                h = max(h,books[j-1][1])
                dp[i] = min(dp[i],dp[j-1]+h)
        return dp[-1]
```

##### 1235. 规划兼职工作

##### 1335. 工作计划的最低难度

##### 1187. 使数组严格递增

##### 514. 自由之路

##### 887. 鸡蛋掉落

> 鸡蛋越多操作数才可能减少，要不然只能逐层尝试。分类讨论从每层扔下鸡蛋摔碎或完好的情况，因为要考虑最坏情况，所以需要选择合适的楼层使两种情况的最大值最小。

* 动态规划+二分法。dp\[i][j]表示i层楼j个鸡蛋的在最坏情况下的最小操作次数。O(knlogn)

  * 不使用第j个鸡蛋：$dp[i][j-1]$

  * 使用第j个鸡蛋，在第x层扔下，选择二者中的较大值也就是最坏情况：

    * 蛋碎，搜索空间变成1~x-1，$dp[x-1][j-1]$
    * 未碎，搜索空间变成x+1~i，$dp[i-x][j]$

    枚举楼层x，找到最小的操作次数后加1。

    * 为了加快找到最合适x的速度，采用二分法。上面两种情况的操作数分别随x单调递增和单调递减，当蛋碎和未碎的操作数接近时，二者的最大值最小。由于其是离散函数，当不存在刚好相等的数据点时，需要比较交点左右两点的操作数。

```python
# 自底向上，速度过慢，计算了无用的子问题。
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0]*(k+1) for _ in range(n+1)]
        for i in range(1,n+1):
            dp[i][1] = i
        for j in range(1,k+1):
            dp[1][j] = 1
        for i in range(2,n+1):
            lim = min(int(math.log(i,2))+1,k)   # 排除鸡蛋过多的情况
            for j in range(2,k+1):
                dp[i][j] = dp[i][j-1]
                if j > lim:
                    continue
                l, r = 1, i
                while l <= r:
                    mid = (l+r)//2
                    if dp[mid-1][j-1]<dp[i-mid][j]:
                    # 蛋碎大于等于未碎的最小x，x-1即为未碎大于等于蛋碎的最大x
                        l = mid + 1             
                    elif dp[mid-1][j-1]>dp[i-mid][j]:
                        r = mid - 1
                    else:
                        l = mid
                        break
                dp[i][j] = min(dp[l-1][j-1]+1,dp[i-l+1][j]+1)   
                '''
                枚举x   超时
                for x in range(1,i+1):
                    dp[i][j] = min(dp[i][j],max(dp[x-1][j-1],dp[i-x][j])+1)
                '''
        return dp[-1][-1]
```

```python
# 自顶向下
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        def recur(k,n):
            if (k,n) not in memo:
                if k == 1 or n<=2:
                    return n
                lim = int(math.log(n,2))+1
                if k >= lim:
                    return lim	# 鸡蛋充足，直接二分查找。
                l, r = 1, n
                while l <= r:	
                    mid = (l+r) // 2
                    t1, t2 = recur(k-1,mid-1),recur(k,n-mid)
                    if t1 < t2:
                        l = mid + 1
                    elif t1 > t2:
                        r = mid - 1
                    else:
                        l = mid 
                        break
                memo[(k,n)] = 1 + min(recur(k-1,l-1),recur(k,n-l+1))
            return memo[(k,n)]
        memo = {}
        return recur(k,n)
```

* 另一种状态描述，dp\[t][k]表示给定k枚鸡蛋和t次操作能检测的建筑高度，找到使dp\[t][k]>=n的最小t。O(kn)
  * dp\[t][k] = 1 + dp\[t-1][k-1] + dp\[t-1][k] ，考虑多出的一个鸡蛋摔碎和未摔碎两种情况，能测的最大高度即为上下两部分高度加上测试层。

```python
class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        if k== 1 or n <= 2:
            return n
        dp = [0] * (k+1)				# 空间优化
        for i in range(1,n+1):			# 逐渐增加操作次数，最多操作n次
            for j in range(k,0,-1):
                dp[j] += 1 + dp[j-1]
            if dp[k] >= n:
                return i
```

#### 回文子序列

##### 516. 最长回文子序列

* dp\[i][j]表示i~j的最长子序列长度。注意遍历的顺序，提前计算所需的状态。

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[1]*n for _ in range(n)]
        for j in range(1,n):
            for i in range(j-1,-1,-1):
                if s[i] == s[j]:
                    dp[i][j] = (dp[i+1][j-1] if i+1<j else 0) + 2
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        return dp[0][-1]
```

##### 5. 最长回文子串

* dp\[i][j]表示字符串是否是回文子串。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False]*n for _ in range(n)]
        res = s[0]
        for j in range(n):
            dp[j][j] = True
            for i in range(j-1,-1,-1):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] if i+1<j else True
                    if dp[i][j] and j-i+1 > len(res):
                        res = s[i:j+1]
        return res
```

##### 131. 分割回文串

* dp记录所有的回文子串，然后利用回溯法枚举所有分割方式。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def dfs(i,cur):
            if i == n:
                res.append(cur)
                return 
            for j in range(i,n):
                if dp[i][j]:
                    dfs(j+1,cur+[s[i:j+1]])

        n = len(s)
        dp = [[False]*n for _ in range(n)]
        for j in range(n):
            dp[j][j] = True
            for i in range(j-1,-1,-1):
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] if i+1<j else True
        res = []
        dfs(0,[])
        return res
```

##### 132. 分割回文串 II

* dp[i]表示s[0:i+1]的最小分割次数。

```python
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        dp = list(range(n))
        for j in range(1,n):  
            if s[:j+1] == s[:j+1][::-1]:
                dp[j] = 0
                continue
            for i in range(j,0,-1):
                if s[i:j+1]==s[i:j+1][::-1]:
                    dp[j] = min(dp[j],dp[i-1]+1)     
        return dp[-1]
```

* 同时记录回文串情况和分割次数。

```python
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        p = [[False]*n for _ in range(n)]
        dp = list(range(n))
        for j in range(1,n):
            p[j][j] = True
            dp[j] = dp[j-1]+1
            for i in range(j-1,-1,-1):
                if s[i]==s[j]:
                    p[i][j] = p[i+1][j-1] if i+1<j else True
                    if p[i][j]:
                        dp[j] = min(dp[j],dp[i-1]+1) if i else 0
        return dp[-1]
```

* 记忆化递归。

```python
class Solution:
    @functools.lru_cache(None)
    def minCut(self, s: str) -> int:
        if s == s[::-1]:
            return 0
        ans = float("inf")
        for i in range(1, len(s) + 1):
            if s[:i] == s[:i][::-1]:
                ans = min(self.minCut(s[i:]) + 1, ans)
        return ans
```

##### 647. 回文子串

##### 730. 统计不同回文子串

##### 1312. 让字符串成为回文串的最少插入次数

#### 最长递增子序列

##### 300. 最长递增子序列   LIS

* 动态规划。dp[i]表示以nums[i]结尾的最长递增子序列的长度。dp[i]既代表以i结尾的状态，同时也是子问题求解的一个阶段。O(n^2)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1]*n
        for i in range(1,n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)
```

* 贪心+二分。单调栈保存现在发现的上升子序列，为了使子序列尽可能的长，需要寻找尽可能小的值放到序列末尾。O(nlogn)

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        seq = []
        for each in nums:
            if not seq or each > seq[-1]:
                seq.append(each)
            else:
            	# 替换掉当前子序列中的较大值,子序列长度不变
                l, r = 0, len(seq)-1
                while l <= r:
                    mid = (l+r)//2
                    if seq[mid] < each:
                        l = mid + 1
                    else:
                        r = mid - 1
                seq[l] = each  # seq中大于等于each的最小值 
        return len(seq)
```

##### 673. 最长递增子序列的个数

* dp\[i][0]，dp\[i][1]分别计算以i结尾的最长递增子序列的长度和数量。

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[1]*2 for _ in range(n)]
        for j in range(1,n):
            for i in range(j):
                if nums[i]<nums[j]:
                    if dp[i][0] >= dp[j][0]:
                        dp[j][0] = dp[i][0] + 1
                        dp[j][1] = dp[i][1]
                    elif dp[i][0] == dp[j][0]-1:
                        dp[j][1] += dp[i][1]
        l = max(each[0] for each in dp)
        return sum(each[1] for each in dp if each[0]==l)
```

##### 1218. 最长定差子序列

##### 334. 递增的三元子序列

* 可以转化为求最长递增子序列的长度，如果长度大于等于3则存在递增的三元子序列。实际上只需要确定长度为3的上升子序列存在，所以用于存储子序列的数组长度只需要2，

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        fir = sec = inf
        for each in nums:
            if each>sec:
                return True
            elif fir<each<sec:
                sec = each
            elif fir>each:
                fir = each
        return False
```

##### 354. 俄罗斯套娃信封问题	

* 动态规划。排序后转化为最长（高度）递增子序列。

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        # 根据宽度递增，宽度相同时根据高度递减
        envelopes.sort(key=lambda x: (x[0],-x[1]))
        dp = [1] * n
        for i in range(1,n):
            for j in range(i):
                if envelopes[j][1]<envelopes[i][1]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)
```

* 贪心+二分。

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0],-x[1]))
        stack = []
        for _,h in envelopes:
            if not stack or stack[-1]<h:
                stack.append(h)
            else:
                l, r = 0, len(stack)-1
                while l <= r:
                    mid = (l+r)//2
                    if stack[mid]<h:
                        l = mid + 1
                    else:
                        r = mid - 1
                stack[l] = h
        return len(stack)
```

##### 1671. 得到山形数组的最少删除次数

* 转化为使顶点左右两侧的最长上升子序列长度和最大。

```python
class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        right = [1]*n	
        cur = [nums[-1]]
        for i in range(n-2,0,-1):
            if nums[i] > cur[-1]:
                cur.append(nums[i])
            else:
                cur[bisect.bisect_left(cur,nums[i])] = nums[i]
            right[i] = len(cur)
        res = 0
        cur = [nums[0]]
        for i in range(1,n-1):
            if nums[i] > cur[-1]:
                cur.append(nums[i])
            else:
                cur[bisect.bisect_left(cur,nums[i])] = nums[i]
            if len(cur)>1 and right[i]>1:
                res = max(res,len(cur)+right[i])
        return n-res+1	# 最大的情况左右两侧序列共用顶点，所以要+1
```

##### 1964. 找出到每个位置为止最长的有效障碍赛跑路线

* 同300。

```python
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        stack = []
        res = []
        for each in obstacles:
            if not stack or each >= stack[-1]:
                stack.append(each)
                res.append(len(stack))
            else:
                idx = bisect.bisect(stack,each)
                stack[idx] = each
                res.append(idx+1)
        return res
```

##### 1626. 无矛盾的最佳球队

##### 5959. 使数组k递增的最少操作次数

* 将原数组划分成k个子数组，只要找到这个数组中最长的递增子序列，然后用总长度减去该子序列长度就是需要调整的元素个数。

```python
class Solution:
    def kIncreasing(self, arr: List[int], k: int) -> int:
        def lis(nums):
            stack = []
            for each in nums:
                if not stack or each >= stack[-1]:
                    stack.append(each)
                else:
                    l, r = 0, len(stack)-1
                    while l <= r:
                        mid = (l+r)//2
                        if stack[mid] <= each:
                            l = mid+1
                        else:
                            r = mid-1
                    stack[l] = each
            return len(nums) - len(stack)

        n = len(arr)
        res = 0
        for i in range(k):
            res += lis(arr[i:n:k])
        return res
```

#### 子数组

##### 152. 乘积最大子数组

* dp[i]记录以i结尾的子数组乘积最大值和最小值。

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[nums[i]]*2 for i in range(n)]
        for i in range(1,n):
            if nums[i] >= 0:
                dp[i][0] = max(nums[i],dp[i-1][0]*nums[i])
                dp[i][1] = min(nums[i],dp[i-1][1]*nums[i])
            else:
                dp[i][0] = max(nums[i],dp[i-1][1]*nums[i])
                dp[i][1] = min(nums[i],dp[i-1][0]*nums[i])
        return max(dp[i][0] for i in range(n))
    
# 优化空间
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_ = max_ = res = nums[0]
        for i in range(1,len(nums)):
            if nums[i] >=0:
                max_, min_ = max(max_*nums[i],nums[i]),min(min_*nums[i],nums[i])
            else:
                max_, min_ = max(min_*nums[i],nums[i]),min(max_*nums[i],nums[i])
            res = max(res,max_)
        return res
```

##### 1186. 删除一次得到子数组的最大和

* dp\[i][0]记录以i结尾的最大子序和，dp\[i][1]记录从以i结尾的子序列中删除一个后得到的最大和。

```python
class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        n = len(arr)
        dp = [[0]*2 for _ in range(n)]
        dp[0][0] = res = arr[0]
        for i in range(1,n):
            dp[i][0] = max(dp[i-1][0]+arr[i],arr[i])
            dp[i][1] = max(dp[i-1][0],dp[i-1][1]+arr[i])
            res = max(res,dp[i][0],dp[i][1])	# 删除或不删除
        return res
```

##### 523. 连续的子数组和

##### 674. 最长连续递增子序列

##### 718. 最长重复子数组

#### 树塔

##### 120. 三角形最小路径和

* dp\[i][j]表示走到第i行第j列的最小路径和。

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [0]+[float('inf')] * n
        for i in range(n):
            for j in range(i,-1,-1):
                dp[j] = min(dp[j],dp[j-1])+triangle[i][j]
        return min(dp)
```

##### 118. 杨辉三角

##### 931. 下降路径最小和

##### 1289. 下降路径最小和2

##### 1301. 最大得分的路径数目

#### 排列组合

##### 698. 划分为k个相等的子集

##### 487. 最大连续1的个数 II

*  dp\[i][0]为以 i 为结尾未使用操作的最大的连续 1 的个数，dp\[i][1]为以 i为结尾使用操作将 [0,i]某个 0 变成 1 的最大的连续 1 的个数。


```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0]*2 for _ in range(n)]
        res = 0
        for i in range(n):
            if nums[i]:
                dp[i][0] = dp[i-1][0] + 1
                dp[i][1] = dp[i-1][1] + 1
            else:
                dp[i][1] = dp[i-1][0] + 1
            res = max(res,dp[i][0],dp[i][1])
        return res
```

* 滑动窗口，维护一个0数量小于2的窗口。

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        i = res = flag = 0
        for j in range(len(nums)):
            if not nums[j]:
                flag += 1
            while flag > 1:
                if not nums[i]:
                    flag -= 1
                i += 1
            res = max(res, j-i+1)
        return res
    
# 处理数据流，队列记录0的位置
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        i = res = cnt = 0
        q = deque([])
        for j in range(len(nums)):
            if not nums[j]:
                q.append(j)
                cnt += 1
            while cnt > 1:
                i = q.popleft()+1
                cnt -= 1
            res = max(res, j-i+1)
        return res
```

##### 920. 播放列表的数量

* dp\[i][j]表示长度为i且包含j首不同歌曲的播放列表的数量。考虑最后一首歌可以选择播放过的或未播放的。

```python
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        dp = [[0]*(n+1) for _ in range(goal+1)]
        dp[0][0] = 1
        mod = 10**9+7
        for i in range(1,goal+1):
            for j in range(1,n+1):
                dp[i][j] = (dp[i-1][j-1]*(n-j+1)+dp[i-1][j]*max(j-k,0))%mod
        return dp[-1][-1]

# 空间优化
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        dp = [0]*(n+1)
        dp[1] = n
        mod = 10**9+7
        for i in range(2,goal+1):
            for j in range(min(i,n),0,-1):
                dp[j] = (dp[j-1]*(n-j+1)+dp[j]*max(j-k,0))%mod
        return dp[-1]
```

##### 276. 栅栏涂色

* dp[i]表示0～i有效涂色的方案数。考虑是否刷与前一个栏杆相同的颜色。

```python
class Solution:
    def numWays(self, n: int, k: int) -> int:
        if n == 1:
            return k
        dp = [0]*n
        dp[0], dp[1] = k, k*k
        for i in range(2,n):
            dp[i] = (dp[i-1] + dp[i-2])*(k-1)
        return dp[-1]
```

##### 940. 不同的子序列 II

* dp[i]表示s[:i+1]含有的不同子序列的个数。记录是否存在重复的字母及其最后出现的位置。

```python
class Solution:
    def distinctSubseqII(self, s: str) -> int:
        dp = [0]*(len(s)+1)
        memo = {}
        mod = 10**9+7
        for i,each in enumerate(s):
            if each not in memo:
                dp[i+1] = (2*dp[i]+1)%mod
            else:
                dp[i+1] = (2*dp[i]-dp[memo[each]])%mod	# 减去重复的组合
            memo[each] = i
        return dp[-1]
```

##### 518. [零钱兑换 II](#jump1)

##### 377. [组合总和 Ⅳ](#jump2)

#### 双序列

一般是二维DP分别表示在两个序列中的位置。

***

##### 14. 最长公共前缀

##### 392. 判断子序列

##### 1143. 最长公共子序列	LCS

* dp\[i][j]表示text1的前i项和text2的前j项的最长公共子序列的长度。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[-1][-1]
```

##### 1035. 不相交的线

##### 1458. 两个子序列的最大点积

##### 1092. 最短公共超序列

* 先求最长公共子序列，dp\[i][j]存储str1[:i]和str2[:j]的最长公共子序列。然后插入两个字符串中不再公共子序列中的字符。

```python
class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        n1, n2 = len(str1), len(str2)
        dp = [['']*(n2+1) for _ in range(n1+1)]
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                if str1[i-1]==str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+str1[i-1]
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1],key=len)
        lcs = dp[-1][-1]
        res = ''
        i = j = 0
        for each in lcs:
            while i < n1 and str1[i] != each:
                res += str1[i]
                i += 1
            while j < n2 and str2[j] != each:
                res += str2[j]
                j += 1
            res += each
            i += 1
            j += 1
        return res+str1[i:]+str2[j:]
```

##### 1713. 得到子序列的最小操作数

* 最长公共子序列超时，由于target中没有重复元素，可以将问题转换成在arr中寻找最长递增子序列。

```python
class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        n = len(target)
        d = dict(zip(target,range(n)))
        stack = []
        for each in arr:
            if each in d:
                i = d[each]
                idx = bisect.bisect_left(stack,i)
                if idx < len(stack):
                    stack[idx] = i
                else:
                    stack.append(i)
        return n-len(stack)
```

##### 72. 编辑距离

* dp\[i][j]表示A的前i个字母和B的前j个字母之间的编辑距离。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1), len(word2)
        dp = [[0]*(n2+1) for _ in range(n1+1)]
        for i in range(1,n1+1):
            dp[i][0] = i
        for j in range(1,n2+1):
            dp[0][j] = j
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1
        return dp[-1][-1]
```

##### 44. 通配符匹配

* dp\[i][j]表示 s 的前 i 个字符与 p 中的前 j 个字符是否能够匹配。

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(m+1):	# i从0开始，因为*可以匹配空字符串。
            for j in range(1,n+1):
                if p[j-1] == '*':
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]
                elif i and (s[i-1]==p[j-1] or p[j-1]=='?'):
                    dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]
```

##### 10. 正则表达式匹配

* dp\[i][j]表示 s 的前 i 个字符与 p 中的前 j 个字符是否能够匹配。

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(m+1):	# i从0开始，因为x*可以匹配空字符串。
            for j in range(1,n+1):
                if p[j-1] == '*':
                    if s[i-1]==p[j-2] or p[j-2]=='.':
                        dp[i][j] = dp[i][j-2] or dp[i][j-1] or dp[i-1][j]
                    else:
                        dp[i][j] = dp[i][j-2]
                elif i and (s[i-1] == p[j-1] or p[j-1]=='.'):
                    dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]
```

##### 97. 交错字符串

* dp\[i][j]记录s1[:i]和s2[:j]能否构成s3[:i+j]。

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        n1, n2, n3 = len(s1), len(s2), len(s3)
        if n1+n2 != n3:
            return False
        dp = [[False]*(n2+1) for _ in range(n1+1)]
        dp[0][0] = True
        for i in range(1,n1+1):
            dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
        for j in range(1,n2+1):
            dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                dp[i][j] = (dp[i-1][j] and s1[i-1]==s3[i+j-1]) or (dp[i][j-1] and s2[j-1]==s3[i+j-1])
        return dp[-1][-1]
```

##### 115. 不同的子序列

* dp\[i][j]表示字符串s的前j个字符中字符串t的前i个字符出现的次数。

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = 1
        for i in range(1,m+1):
            for j in range(1,n+1):
                if s[i-1]==t[j-1]:
                    dp[i][j] = dp[i-1][j-1]+dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[-1][-1]
```

##### 583. 两个字符串的删除操作

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1, n2 = len(word1), len(word2)
        dp = [[0]*(n2+1) for _ in range(n1+1)]
        for i in range(1,n1+1):
            dp[i][0] = i
        for j in range(1,n2+1):
            dp[0][j] = j
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i][j-1])+1
        return dp[-1][-1]
```

##### 712. 两个字符串的最小ASC II删除和

##### 1216. 验证回文字符串 III

* 转化为LCS。计算s和s[::-1]的最长公共子序列。

```python
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        n = len(s)
        t = s[::-1]
        dp = [[0]*(n+1) for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,n+1):   
                if s[i-1] == t[j-1]:             
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1],dp[i-1][j])
        return n-dp[-1][-1]<=k
```

##### 467. 环绕字符串中唯一的子字符串

##### 678. 有效的括号字符串

##### 639. 解码方法2

##### 1320. 二指输入的最小距离

#### 博弈

##### 375. 猜数字大小 II

* min-max问题。dp\[i][j]表示在范围i~j内确定胜利的最少金额，从最小的子问题开始计算。

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(n+1)]
        for i in range(n-1,-1,-1):
            for j in range(i+1,n+1):
                dp[i][j] = min(max(dp[i][k-1],dp[k+1][j])+k for k in range(i,j))
        return dp[1][n]
```

##### 292. Nim游戏

##### 1025. 除数博弈

##### 464. 我能赢吗

##### 486. 预测赢家

##### 877. 石子游戏

##### 1140. 石子游戏2

##### 1406. 石子游戏3

##### 1510. 石子游戏4

#### 其它

##### 32. 最长有效括号

##### 943. 最短超级串

##### 1531. 压缩字符串 II

* dp\[i][j]表示从前i个字符中最多删除j个字符后的最短长度。

```python
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        def compress(x):	# 压缩x个相同字符
            return 1 if x <= 1 else 2 if 1<x<10 else 3 if 10<=x<100 else 4

        n = len(s)
        dp = [[i]*(k+1) for i in range(n+1)]
        for i in range(1,n+1):
            for j in range(min(i+1,k+1)):
                if j != 0:
                    dp[i][j] = dp[i-1][j-1]    # 删除第i个字符
                # 不删除i时，为了获得可以压缩的子串，在i之前需要删除的字符数量
                cnt = 0     
                for p in range(i,0,-1):        # 遍历可能的压缩方式
                    if s[i-1] != s[p-1]:
                        cnt += 1
                        if cnt > j:     # 最多删除j个
                            break
                    else:
                        dp[i][j] = min(dp[p-1][j-cnt]+compress(i-p+1-cnt),dp[i][j])
        return dp[-1][-1]
```

##### 741. 摘樱桃

* 从左上角要找到两条到达右下角的路径，将路径的长度作为DP的阶段，每个阶段两条路径同时扩展1。因为阶段不足以表示状态，需要额外的维度来描述状态，即两条路径的末尾位置。dp\[i]\[x1][x2]分别表示路径长度和两条路径末尾的行数。

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        n = len(grid)
        dp = [[float('-inf')]*n for _ in range(n)]
        dp[0][0] = grid[0][0]
        for step in range(1,2*n-1):
            new = [[float('-inf')]*n for _ in range(n)]
            for i in range(max(0,step-n+1),min(n-1,step)+1):
                for j in range(i,min(n-1,step)+1):  # 剪枝
                    if grid[i][step-i]<0 or grid[j][step-j]<0:
                        continue
                    tmp = grid[i][step-i]
                    if i != j:
                        tmp += grid[j][step-j]
                    # 上一个格子在左侧或上方，选择最大值。
                    new[i][j] = max(dp[x][y] for x in (i-1,i) for y in (j-1,j) if x >= 0 and y >= 0)+tmp
            dp = new
        return max(0,dp[-1][-1])
```

##### 1463. 摘樱桃 II

* 摘到每一行是一个阶段，两个机器人在每一行的位置构成具体的状态。

```python
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[float('-inf')]*n for _ in range(n)]
        dp[0][-1] = grid[0][0] + grid[0][-1]
        for step in range(1,m):
            new = [[float('-inf')]*n for _ in range(n)]
            for i in range(n):
                for j in range(i,n):
                    tmp = grid[step][i] + (grid[step][j] if i != j else 0)
                    new[i][j] = max(dp[x][y] for x in [i-1,i,i+1] for y in [j-1,j,j+1] if 0<=x<n and 0<=y<n)+tmp
            dp = new 
        return max(max(each) for each in dp)
```

##### [杨老师的照相排列](https://www.acwing.com/problem/content/273/)

* bfs，先定义状态，再通过当前状态应该更新哪些未知状态给出状态转移方程。dp[(a1,a2,...,a3)]表示每行分别站了a1,a2,...,a3个学生时排列次数。

```python
from collections import defaultdict

class Solution:
    def permutation(self,m,rows):
        if m == 1:
            return 1
        dp = defaultdict(int)
        dp[(0,)*m] = 1
        for _ in range(sum(rows)):	# 逐个放置所有学生
            new = defaultdict(int)
            for state,val in dp.items():
                for i in range(m):
                    if state[i] == rows[i]:
                        continue
                    if not i or state[i]<state[i-1]:
                        cur = list(state)
                        cur[i] += 1
                        cur = tuple(cur)
                        new[cur] += dp[state]
            dp = new
        return dp[tuple(rows)]

if __name__ == '__main__':
    solution = Solution()
    while True:
        try:
            m = int(input())
            rows = list(map(int,input().split()))
            print(solution.permutation(m,rows))
        except EOFError:
            break
```

##### [最长公共上升子序列](https://www.acwing.com/problem/content/274/)

* dp\[i][j]表示A的前i项和B的前j项构成的**以B[j-1]结尾**的最长公共上升子序列的长度。决策集合只增大不减小时，可以维护一个决策集合的最优值做到O(1)的决策。

```python
class Solution:
    def maxCommonUperString(self,n,A,B):
        dp = [[0]*(n+1) for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,n+1):
                if A[i-1] != B[j-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = 1
                    for k in range(1,j):	# 找到满足上升条件的前缀子序列
                        if B[k-1] < A[i-1]:
                            dp[i][j] = max(dp[i][j],dp[i-1][k]+1)
        return max(dp[-1])
    
# 优化 k构成的决策集合在进行选择时，对不同的j来说存在重复情况，只需要两次遍历即可。
class Solution:
    def maxCommonUperString(self,n,A,B):
        dp = [[0]*(n+1) for _ in range(n+1)]
        for i in range(1,n+1):
            cur = 0     # 记录决策集合中的最大值，不需要再进行遍历。
            for j in range(1,n+1):
                dp[i][j] = cur+1 if A[i-1] == B[j-1] else dp[i-1][j]
                if A[i-1] > B[j-1]:		# 满足上升条件
                    cur = max(cur,dp[i-1][j])
        return max(dp[-1])
    
if __name__ == '__main__':
    n = int(input())
    A = list(map(int,input().split()))
    B = list(map(int,input().split()))
    solution = Solution()
    print(solution.maxCommonUperString(n,A,B))
```

##### [分级](https://www.acwing.com/problem/content/275/)

* dp\[i][j]表示构造长度为i的序列B，最后一项为j时，与A的前i项差的最小值。dp\[i][j]=dp\[i-1][k] (k in [1,j]) + |j-Ai|。
  * 一定存在一个最优解B，其中的元素都在A中出现过，从而将j限制在set(A)中。

```python
class Solution:
    def makegrade(self,A,n):
        nums = sorted(list(set(A)))
        m = len(nums)
        dp = [[0]*m for _ in range(n+1)]
        for i in range(1,n+1):
            diff = float('inf')		# 保存决策集合中的最小值
            for j in range(m):
                diff = min(diff,dp[i-1][j])		# 前一项的取值小于等于j
                dp[i][j] = diff+abs(A[i-1]-nums[j])
        return min(dp[-1])

if __name__ == '__main__':
    solution = Solution()
    n = int(input())
    A = [int(input()) for _ in range(n)]
    # A逆序模拟非严格单调减
    print(min(solution.makegrade(A,n),solution.makegrade(A[::-1],n)))
```

##### [饼干](https://www.acwing.com/problem/content/279/)

* 怨气值大的孩子分配较多的饼干，对孩子按怨气从大到小进行排序，按照递减的顺序分配饼干，dp\[i][j]表示前i个孩子分配j个饼干时的最小怨气值。按照划分方案中已确定获得1个饼干的孩子数量，该集合可以分成i个子集。

```python
# 超时  
class Solution:
    def cookies(self,g,n,m):
        child = list(range(n))
        child.sort(key=lambda x: -g[x])
        prefix = [0]
        for i in range(1,n+1):
            prefix.append(prefix[-1]+g[child[n-i]])
        dp = [[float('inf')]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = 0
        dp[1] = [0] * (m+1)
        for i in range(2,n+1):
            for j in range(i,m+1):
                dp[i][j] = dp[i][j-i]       # 每个孩子都多发一个饼干，大小关系不变。
                for k in range(1,i+1):      # 有k个孩子只有1个饼干
                    dp[i][j] = min(dp[i][j],dp[i-k][j-k]+(i-k)*(prefix[k+n-i]-prefix[n-i]))
        i,j,h = n,m,0
        res = [0] * n
        while i and j:                          # 逆推每个孩子发的饼干数量
            if j > i and dp[i][j]==dp[i][j-i]:
                j -= i
                h += 1                      # 前i个孩子发的饼干数
            else:
                for k in range(1,min(i,j)+1):    # 找到分发方案
                    if dp[i][j] == dp[i-k][j-k]+(i-k)*prefix[k]:
                        for u in range(i-k+1,i+1):
                            res[child[u-1]] = h+1
                        i -= k
                        j -= k
                        break
        return str(dp[-1][-1])+'\n'+' '.join(map(str,res))

if __name__=='__main__':
    n, m = map(int,input().split())
    g = list(map(int,input().split()))
    solu = Solution()
    print(solu.cookies(g,n,m))
```



### 背包

背包问题属于线性DP中一类特殊的模型。

#### 01背包

> 有$n$个物品和一个容量为$W$的背包，每个物品有重量$w_i$和价值$v_i$两种属性，要求选若干物品放入背包使背包中物品的总价值最大且背包中物品的总重量不超过背包的容量。

将“已处理的物品数”作为DP的“阶段”，以“背包中已经放入的重量”作为附加维度。dp\[i][j]表示**只能放前 i个物品的情况下，容量为j的背包所能达到的最大总价值**，根据**是否选择第i个物品(0或1)**划分子问题：
$$
dp[i][j] = max\left\{\begin{matrix}dp[i-1][j]\\dp[i-1][j-w_i]+v_i \end{matrix}\right.
$$
每一阶段i的状态只与上一阶段i-1的状态有关，采用滚动数组的形式优化内存:
$$
dp[i] = max(dp[j],dp[j-w_i]+v_i)
$$
注意在枚举的过程中需要**逆序**进行，因为第j项需要利用到j-w项更新前的值。

```python
for i in range(n):
    for j in range(W,wi-1,-1):
        dp[j] = max(dp[j],dp[j-w[i]]+v[i])
res = max(dp)
```

* 该类型的核心是单个物品的取舍，目标不一定是求最值。
* 不是刚好放置i个物品，价值达到j，而是处于这个区间内。

##### 416. 分割等和子集

* 是否能使总价值刚好为该值，dp[i]表示能否得到元素和为i。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if s&1:
            return False
        t = s // 2
        dp = [True]+[False]*t
        for each in nums:
            for i in range(t,each-1,-1):
                dp[i] |= dp[i-each]
        return dp[-1] 
```

##### 494. 目标和

* 求方案数。x+y=sum, x-y=target，从而推出背包中需要装入的数字和。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        t = target+sum(nums)
        if t<0 or t%2:
            return 0
        t //= 2
        dp = [1]+[0]*t
        for each in nums:
            for i in range(t,each-1,-1):
                dp[i] += dp[i-each]
        return dp[-1]
```

##### 1049. 最后一块石头的重量 II

* 重量同时也是价值，背包容量为1/2总重量，求最大重量。dp[i]表示能否容纳重量为i的石头。

```python
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        t = sum(stones)//2
        dp = [True]+[False]*t
        for each in stones:
            for i in range(t,each-1,-1):
                dp[i] |= dp[i-each]
        for i in range(t,-1,-1):
            if dp[i]:
                return sum(stones)-2*i
```

##### 956. 最高的广告牌

* 将两根钢筋的差值作为“背包”中物品的价值，但背包容量不限。dp{i:j}记录钢筋的第一根钢筋和第二根钢筋的差值i和第一根钢筋的高度j，希望在相同的高度差下，第一根钢筋尽可能高。

```python
class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        dp = {0:0}							# 用字典存储状态
        for each in rods:
            for k,v in list(dp.items()):	# 注意是未更新前的字典值
                dp[k+each] = max(dp.get(k+each,0),v+each)
                dp[k-each] = max(dp.get(k-each,0),v)
        return dp[0]
```

##### 474. 一和零

* 二维背包，用二维数组来表示当前价值，dp\[i][j]表示**不超过**i个0、j个1时的最大长度。

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(m+1)]
        for each in strs:
            zero, one = each.count('0'), each.count('1')
            for i in range(m,zero-1,-1):
                for j in range(n,one-1,-1):
                    dp[i][j] = max(dp[i][j],dp[i-zero][j-one]+1)
        return dp[m][n]
```

##### 879. 盈利计划

* 将任务看作“石头”，“石头”的质量由参与人数和利润两项决定，背包大小受到上限（任务数量）和下限（最少利润）的双重约束，因此是二维背包，dp\[i][j]表示利润**至少**为i，人数不超过j时的计划数。

```python
class Solution:
    def profitableSchemes(self, n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
        dp = [[0]*(n+1) for _ in range(minProfit+1)]
        for j in range(n+1):    # 没有任务时，利润为0，方案数为1
            dp[0][j] = 1 
        mod = 10**9+7
        for x,y in zip(profit,group):
            for i in range(minProfit,-1,-1):	# 因为是“至少”，注意下限是0
                for j in range(n,y-1,-1):
                    dp[i][j] += dp[max(i-x,0)][j-y] % mod
        return dp[-1][-1] % mod
```

##### [背包问题求具体方案](https://www.acwing.com/problem/content/description/12/)

* 因为要选择字典序最小的方案，所以进行DP时要从后往前递推，以便回溯选择时正序进行。

```python
class Soulution:
    def bag(self,n,v,l):
        dp = [[0]*(v+1) for _ in range(n+1)]
        for i in range(n-1,-1,-1):
            a, b = l[i]
            for j in range(1,v+1):
                dp[i][j] = dp[i+1][j]		# 注意需要记录没空间且不选当前物品的情况
                if j >= a:
                    dp[i][j] = max(dp[i+1][j],dp[i+1][j-a]+b)
        res = []
        cur = v
        for i in range(n):
            if cur >= l[i][0] and dp[i][cur] == dp[i+1][cur-l[i][0]]+l[i][1]:
                cur -= l[i][0]
                res.append(i+1)
        return res



if __name__=='__main__':
    n, v = map(int,input().split())
    l = []
    for _ in range(n):
        l.append(tuple(map(int,input().split())))
    solu = Soulution()
    print(*solu.bag(n,v,l))
```

##### [数字组合](https://www.acwing.com/problem/content/description/280/)

* 求和类型。

```python
class Solution:
    def cnt(self,m,nums):
        dp = [1]+[0]*m
        for each in nums:
            for i in range(m,each-1,-1):
                dp[i] += dp[i-each]
        return dp[-1]

if __name__ == '__main__':
    _, m = map(int,input().split())
    nums = list(map(int,input().split()))
    solu = Solution()
    print(solu.cnt(m,nums))
```

##### [陪审团](https://www.acwing.com/problem/content/description/282/)

* 

```python

```

#### 完全背包

> 有$n$种物品和一个容量为$W$的背包，每个物品有重量$w_i$和价值$v_i$两种属性，并且有无数个，要求选若干物品放入背包使背包中物品的总价值最大且背包中物品的总重量不超过背包的容量。

dp\[i][j]表示从前i种物品中选出总质量为j的物品放入背包时物品的最大价值和，同样根据是否选择第i个物品划分子问题：
$$
dp[i][j] = max\left\{\begin{matrix}dp[i-1][j]\\dp[i][j-w_i]+v_i \end{matrix}\right.
$$

每一阶段i的状态只与上一阶段i-1的状态有关，采用滚动数组的形式优化内存:
$$
dp[i] = max(dp[j],dp[j-w_i]+v_i)
$$


与01背包不同的是枚举是**正序**的，因为需要在同处于i阶段之间的状态进行转移。									

##### 322. 零钱兑换

* dp[i]表示凑齐金额i需要的硬币个数。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0]+[float('inf')]*amount
        for each in coins:
            for i in range(each,amount+1):
                dp[i] = min(dp[i],dp[i-each]+1)
        return dp[-1] if dp[-1]!=float('inf') else -1
```

* bfs

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        q = deque([amount])
        visited = set()
        cnt = 0
        while q:
            for _ in range(len(q)):
                cur = q.popleft()
                if cur == 0:
                    return cnt
                for each in coins:
                    if cur>=each and cur-each not in visited:
                        q.append(cur-each)
                        visited.add(cur-each)
            cnt += 1
        return -1
```

##### <span id="jump1">518. 零钱兑换 II</span>

* 统计**组合**方法数，dp[i]表示和为i时的方法数。

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1]+[0]*amount
        for each in coins:
            for i in range(each,amount+1):
                dp[i] += dp[i-each]
        return dp[-1]
```

##### <span id="jump2">377. 组合总和 Ⅳ</span>

* 统计**排列**方法数，dp[i]表示和为i时的方法数。但注意排列时物品在内循环，阶段由数值总和划分。

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1]+[0]*target
        nums.sort()
        for i in range(1,target+1):
            for each in nums:
                if each>i:
                    break
                dp[i] += dp[i-each]
        return dp[-1]
```

##### <span id="jump3">638. 大礼包</span>

* dfs，记忆化搜索更易解决。

```python
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(cur):
            if not any(cur):
                return 0
            res = sum(price[i]*cur[i] for i in range(n))
            for each in filter_special:
                nxt = tuple(cur[i]-each[i] for i in range(n))
                if min(nxt) >= 0:
                    res = min(res,each[-1]+dfs(nxt))
            return res
        
        filter_special = []
        for sp in special:
            if sum(sp[i] for i in range(n)) > 0 and sum(sp[i] * price[i] for i in range(n)) > sp[-1]:
                filter_special.append(sp)
        n = len(price)
        return dfs(tuple(needs))
```

* 状态压缩dp，每个状态定义为$dp(i,j_0,j_1,...,j_{n-1})$，即考虑前i个礼包，物品购买j个时的最小花费，采用排列数对状态维度进行压缩，将状态简化为十进制整数。`超时 待优化`

```python
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        n = len(price)
        g = [1] * (n+1)
        # 构造排列数，长度为n，每位进制不同，代表该物品需要买的数量。
        for i in range(1,n+1):
            g[i] = g[i-1]*(needs[i-1]+1)    # 保存和十进制的映射关系
        dp =  [0]+ [inf] * (g[n]-1)               # 转化为一维完全背包
        # 必须遍历状态才能同时处理单独购买和礼包购买两种情况
        for state in range(1,g [n]):       
            cur = [0]*n
            for i in range(n):
                cur[i] = state % g[i+1] // g[i]   # 将状态转换回数组形式
            for i in range(n):                    # 单独购买物品
                if cur[i]>0:
                    dp[state] = min(dp[state],dp[state-g[i]]+price[i])
            for each in special:                  # 购买礼包
                tmp = state
                flag = False
                for i in range(n):
                    if cur[i]<each[i]:
                        flag = True
                        break
                    tmp -= each[i] * g[i]
                if flag:
                    continue
                dp[state] = min(dp[state],dp[tmp]+each[n])
        return dp[-1]
```

##### 1449. 数位成本和为目标值的最大数字

求背包容量为 target，物品重量为cost[i]，价值为 1的完全背包问题。dp\[i][j]表示使用前i个数、成本和恰好为j。

* 记录过程中选择的数字。

```python
class Solution:
    def largestNumber(self, cost: List[int], target: int) -> str:
        # 排除花费中的重复项和过大项
        c = Counter(cost)
        costs = []
        for i,each in enumerate(cost,1):
            if c[each]==1 and each <= target:
                costs.append((i,each))
            else:
                c[each]-=1
        # 在数字相同的情况下较小数字优先放置在低位
        costs = sorted(d.items(),key=lambda x:x[1])	
        dp = ['']+['0']*target
        for c,i in costs:
            for j in range(c,target+1):
                if dp[j-c]!='0':
                    dp[j] = max(dp[j],str(i)+dp[j-c],key=lambda x: (len(x),x))
        return dp[-1]      
```

* 根据结果倒推。

```python
class Solution:
    def largestNumber(self, cost: List[int], target: int) -> str:
        c = Counter(cost)
        costs = []
        for i,each in enumerate(cost,1):
            if c[each]==1 and each <= target:
                costs.append((i,each))
            else:
                c[each]-=1
        dp = [0]+[float('-inf')]*target
        for i,c in costs:
            for j in range(c,target+1):
                dp[j] = max(dp[j],dp[j-c]+1)
        if dp[-1]<0:
            return '0'
        cur = target
        res = []
        for i,c in costs[::-1]:
            while cur>=c and dp[cur] == dp[cur-c]+1:
                res.append(str(i))
                cur -= c
        return ''.join(res)
```



### 区间DP

区间DP属于线性DP的一种，以区间长度作为DP的阶段，使用两个端点坐标描述每个维度，每个状态由若干比它小且包含于它的状态转移而来。

##### 87.扰乱字符串

* 

```pyhton
```



##### 221. 最大正方形

* 由于正方形的规则性可以采用动态规划，dp\[i][j]表示以matrix(i-1,j-1)为右下角的正方形的最大边长，被最近的三个正方形的最短边约束。

```python
# 简化为一维dp
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        res = nw = 0
        dp = [0]*(n+1)
        for i in range(m):
            for j in range(1,n+1):
                dp[j], nw = min(dp[j],dp[j-1],nw)+1 if matrix[i][j-1]=='1' else 0,dp[j]
                res = max(dp[j],res)
        return res*res
```

### 状态压缩DP

##### 1434. 每个人戴不同帽子的方案数

##### 638. [大礼包](#jump3)

