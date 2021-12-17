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

#### 简单转移

利用数据中的前后缀关系，从一个状态转移到某个维度上的相邻状态，如dp[i-1] --> dp[i] ，dp[j] --> dp[i] (j<i) ，一般取最值或求和。

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



***

**LIS**

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



***
**排列组合**

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

***
**交替和**

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

***

**子段和**

##### 53. 最大子序和

* dp[i]表示以i结尾的最大子序和。

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



#### 双序列

一般是二维DP分别表示在两个序列中的位置。

***

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

#### 其它

较为复杂的状态划分和转移方式。

***

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

##### 956. 最高的广告牌

* dp{i:j}记录钢筋的第一根钢筋和第二根钢筋的差值i和第一根钢筋的高度j，希望在相同的高度差下，第一根钢筋尽可能高。

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

##### 杨老师的照相排列

https://www.acwing.com/problem/content/273/

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

##### 最长公共上升子序列

https://www.acwing.com/problem/content/274/

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

##### 分级

https://www.acwing.com/problem/content/275/

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

##### 饼干

https://www.acwing.com/problem/content/279/

* 怨气值大的孩子分配较多的饼干，对孩子按怨气从大到小进行排序，按照递减的顺序分配饼干，dp\[i][j]表示前i个孩子分配j个饼干时的最小怨气值。按照划分方案中已确定获得1个饼干的孩子数量，该集合可以分成i个子集。

```python
# 超时  逆推时间过长
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

#### 01背包

##### 416. 分割等和子集

* dp[i]表示能否得到元素和为i。

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



### 状态压缩DP

##### 1434. 每个人戴不同帽子的方案数

