## 哈希表



#### 1995. 统计特殊四元组

* 将元素两两组合考虑，寻找满足a+b=d-c的四元组。逆序遍历b的同时，哈希表记录d-c取值的情况。

```python
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        cnt = defaultdict(int)
        res = 0
        for b in range(n-3,-1,-1):
            for d in range(b+2,n):
                cnt[nums[d]-nums[b+1]] += 1
            for a in range(b):
                res += cnt[nums[a]+nums[b]]
        return res
```



### 字符串哈希

> 对一个字符串进行比较的时候，时间复杂度为O(N)，但是转为hash值的时候则为O(1)。为避免出现哈希冲突，mod应为足够大的质数。

* 将任意长度字符串映射成一个非负整数

取一固定值P（131，13331），把字符串看作P进制数，并给每种字符分配一个大于0的数值，一般来说分配的数值都远小于P。取一固定值M（2^64），求出该P进制数对M的余数，作为该字符串的哈希值。

在字符串S后添加一个字符c构成的新字符串的hash值是H(S+c) = (H(S)*P+value[c]) mod M；字符串S+T的哈希值为H(S+T) 时，字符串T的哈希值H(T) = (H(S+T)-H(S)\*P^(len(T))) mod M。根据这两种操作即可在O(N)的时间与处理字符串所有前缀哈希值，并在O(1)的时间内查询任意子串的哈希值。

#### 187. 重复的DNA序列

* 字符串中只有4个字母，可以利用二进制代码推导字符串的哈希值。

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        n = len(s)
        if n<=10:
            return []
        d = {'A':0,'C':1,'G':2,'T':3}		# A-->00 C-->01 G-->10 T-->11
        res = []
        cur = 0
        cnt = defaultdict(int)
        for each in s[:10]:
            cur = (cur<<2) | d[each]
        cnt[cur] += 1
        for i in range(n-10):
            cur = ((cur<<2)|d[s[i+10]])&((1<<20)-1)		# 只取有意义的低20位
            cnt[cur] += 1
            if cnt[cur] == 2:
                res.append(s[i+1:i+11])
        return res
```

#### 1044. 最长重复子串

* 哈希+二分。在二分前先通过 O(n)的复杂度预处理出哈希数组，从而确保能够在 check 时能够 O(1)得到某个子串的哈希值，二分检测可能的子串长度。

```python
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        def check(l):
            visited = set()
            for i in range(n-l+1):
                cur = (h[i+l]-h[i]*p[l])%MOD
                if cur in visited:
                    return s[i:i+l]
                visited.add(cur)
            return ''

        MOD = int(1e11 + 7)     # 在c++中可以取2^64，利用usigned longlong类型溢出取模
        P = 13331
        n = len(s)
        h = [0]*(n+1)   # 哈希数组
        p = [1]*(n+1)   # 次方数组
        for i,each in enumerate(s):
            h[i+1] = (h[i]*P+ord(each))%MOD
            p[i+1] = p[i]*P%MOD     # 统计p的n次方，用mod求余化简。
        res = ''
        l, r = 1, n-1
        i = 0
        while l<=r:
            mid = (l+r)//2
            cur = check(mid)
            if cur:
                l = mid + 1
            else:
                r = mid - 1
            if len(cur)>len(res):
                res = cur
        return res
```

