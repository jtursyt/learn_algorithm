## 排序

<table class="tg">
  <tr>
    <th>排序算法</th>
    <th>平均时间复杂度</th>
    <th>最好/最坏</th>
    <th>空间复杂度</th>
    <th>排序方式</th>
    <th>稳定性</th>
  </tr>
  <tr>
    <td>冒泡排序</td>
    <td>O(n2)</td>
    <td>O(n),O(n2)</td>
    <td>O(1)</td>
    <td>in-place</td>
    <td>稳定</td>
  </tr>
  <tr>
    <td>选择排序</td>
    <td>O(n2)</td>
    <td>O(n2),O(n2)</td>
    <td>O(1)</td>
    <td>in-place</td>
    <td>不稳定</td>
  </tr>
  <tr>
    <td>插入排序</td>
    <td>O(n2)</td>
    <td>O(n),O(n2)</td>
    <td>O(1)</td>
    <td>in-place</td>
    <td>稳定</td>
  </tr>
  <tr>
    <td>希尔排序</td>
    <td>O(n^1.5)</td>
    <td>O(nlog^2n),O(n^2)</td>
    <td>O(1)</td>
    <td>in-place</td>
    <td>不稳定</td>
  </tr>
  <tr>
    <td>归并排序</td>
    <td>O(nlogn)</td>
    <td>O(nlogn),O(nlogn)</td>
    <td>O(n)</td>
    <td>out-place</td>
    <td>稳定</td>
  </tr>
  <tr>
    <td>快速排序</td>
    <td>O(nlogn)</td>
    <td>O(nlogn),O(n2)</td>
    <td>O(logn)</td>
    <td>in-place</td>
    <td>不稳定</td>
  </tr>
  <tr>
    <td>堆排序</td>
    <td>O(nlogn)</td>
    <td>O(nlogn),O(nlogn)</td>
    <td>O(1)</td>
    <td>in-place</td>
    <td>不稳定</td>
  </tr>
  <tr>
    <td>计数排序</td>
    <td>O(n+k)</td>
    <td>O(n+k),O(n+k)</td>
    <td>O(k)</td>
    <td>out-place</td>
    <td>稳定</td>
  </tr>
  <tr>
    <td>桶排序</td>
    <td>O(n+k)</td>
    <td>O(n+k),O(n2)</td>
    <td>O(n+k)</td>
    <td>out-place</td>
    <td>稳定</td>
  </tr>
  <tr>
    <td>基数排序</td>
    <td>O(n*k)</td>
    <td>O(n*k),O(n*k)</td>
    <td>O(n+k)</td>
    <td>out-place</td>
    <td>稳定</td>
  </tr>
</table>

> n: 数据规模，k: 桶数，in-place：占用常数内存，不占用额外内存
>
> 稳定：冒泡，插入，归并，桶
>
> 不稳定：选择，希尔，快速，堆

### 冒泡排序

* 重复地访问要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。尾部是已经排好序的元素。当输入已经是正序时最快，当输入是逆序时最慢。

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):		# 已排好序的元素数量
        for j in range(n-i-1):
            if arr[j+1] < arr[j]:
                arr[j+1], arr[j] = arr[j], arr[j+1]
```

### 选择排序

* 在未排序数列中不断的找出最小的元素放到数组前方，头部是已经排好序的元素。

```python
def select_sort(arr):
    n = len(arr)
    for i in range(n):	# 待排序元素起点
        min_idx = i		
        for j in range(i+1,n):	# 找到最小元素
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
```

### 插入排序

* 构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

```python
def insert_sort(arr):
    for i in range(len(arr)):
        for j in range(i-1,-1,-1):
            if arr[j]<arr[j+1]:		# 找到合适的位置
                break
            arr[j], arr[j+1] = arr[j+1], arr[j]
```

```python
def insert_sort(arr):
    for i in range(len(arr)):
        cur = arr[i]
        j = i-1
        while j >= 0 and arr[j] > cur:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = cur       
```

#### 链表排序

##### 147. 对链表进行插入排序

```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        dummy = ListNode(None,head)
        end,cur = head, head.next
        while cur:
            if cur.val < end.val:
                node = dummy
                end.next = cur.next           	   # 删除原节点
                while node.next.val <= cur.val:    # 寻找插入位置
                    node = node.next
                cur.next = node.next
                node.next = cur
            else:
                end = cur
            cur = end.next
        return dummy.next
```

### 希尔排序

* 希尔排序是把记录按下标的一定增量分组，对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的关键词越来越多，当增量减至1时，整个文件恰被分成一组，算法便终止。解决插入排序数据每次只能移动一位的问题。

```python
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap,n):	# 进行直接插入排序
            for j in range(i-gap,-1,-gap):
                if arr[j] < arr[j+gap]:
                    break
                arr[j], arr[j+gap] = arr[j+gap], arr[j]
        gap //= 2
```

### 归并排序

* 归并排序的性能不受输入数据的影响，但需要O(n)的额外空间。递归实现采用分治的思想，每一层分为三步：

  * 分解：将n个元素分成含n/2个元素的子序列。
  * 解决：用合并排序法对两个子序列递归地排序。
  * 合并：合并两个已排序的子序列以得到排序结果。

  利用一个数保存两个数的[方法](https://zhuanlan.zhihu.com/p/159446051)（arr[i] = arr[i] + arr[j]*big_num) 可以实现原地归并，将空间复杂度降至O(1)。

```python
# 递归
def merge_sort(l,r):
	if l < r:
        mid = (l+r)//2
        merge_sort(l,mid)
        merge_sort(mid+1,r)
        left = deque(arr[l:mid+1])
        right = deque(arr[mid+1:r+1])
        merged = [min(left,right).popleft() for _ in range(r-l+1) if min(left,right)]
        merged.extend(list(left) if left else list(right))
        arr[l:r+1] = merged[:]
        
merge_sort(0,len(arr))
```

```python
# 迭代
def merge_sort(arr):
    n = len(arr)
    length = 1	# 从长度为1的数组开始两两合并
    while length < n:
        for left_start in range(0, n-length, 2*length):
            left_end = right_start = left_start+length
            right_end = min(n,right_start+length)
            left = deque(arr[left_start:left_end])
            right = deque(arr[right_start:right_end])
            merged = [min(left,right).popleft() for _ in range(right_end-left_start) if min(left,right)]
            merged.extend(list(left) if left else list(right))
            arr[left_start:right_end] = merged
    	length *= 2
```

#### 链表排序

##### 21. 合并两个有序链表

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        cur = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 if l1 else l2
        return dummy.next
```

##### 23. 合并k个升序链表

```python
# 两两合并
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merge(l1,l2):
            dummy = ListNode()
            cur = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    cur.next = l1
                    l1 = l1.next
                else:
                    cur.next = l2
                    l2 = l2.next
                cur = cur.next
            cur.next = l1 if l1 else l2
            return dummy.next
        
        if not lists:
            return None
        length = 1
        n = len(lists)
        while length<n:			# 分治
            for i in range(0,n-length,2*length):
                lists[i] = merge(lists[i],lists[i+length])
            length *= 2
        return lists[0]
```

```python
# 多路归并
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def __lt__(self,node):		# 重写__lt__
            return self.val < node.val
        ListNode.__lt__ = __lt__
        heap = []					# 优先队列
        for each in lists:
            if each:
                heapq.heappush(heap,each)
        dummy = ListNode()
        cur = dummy
        while heap:
            cur.next = heapq.heappop(heap)
            cur = cur.next
            if cur.next:
                heapq.heappush(heap,cur.next)
        return dummy.next
```

##### 148. 排序链表

```python
# 自顶向下
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def merge(l1,l2):
            dummy = ListNode()
            cur = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    cur.next = l1
                    l1 = l1.next
                else:
                    cur.next = l2
                    l2 = l2.next
                cur = cur.next
            cur.next = l1 if l1 else l2
            return dummy.next

        def partion(head,tail):
            if not head:
                return None
            if head.next == tail:	
                head.next = None	# 返回单个节点
                return head
            fast = slow = head
            while fast!=tail and fast.next!=tail:
                fast = fast.next.next
                slow = slow.next
            return merge(partion(head,slow),partion(slow,tail))

        return partion(head,None)
```

```python
# 自底向上	减少空间复杂度
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def merge(l1,l2):
            dummy = ListNode()
            cur = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    cur.next = l1
                    l1 = l1.next
                else:
                    cur.next = l2
                    l2 = l2.next
                cur = cur.next
            cur.next = l1 if l1 else l2
            return dummy.next

        if not head:
            return None
        cnt, cur = 0, head
        while cur:
            cur = cur.next
            cnt += 1
        dummy = ListNode(None,head)
        length = 1
        while length < cnt:
            pre, cur = dummy, dummy.next
            while cur:
                head1 = cur
                for _ in range(1,length):
                    if cur.next:
                        cur = cur.next
                    else:
                        break
                if not cur.next:
                    pre.next = head1  # 不进行合并直接连接
                    break
                head2 = cur.next
                cur.next = None     # 切分链表
                cur = head2
                for _ in range(1,length):
                    if cur.next:
                        cur = cur.next
                    else:
                        break
                cur.next, cur = None, cur.next  # 切分链表
                pre.next = merge(head1,head2)   # 合并两个链表并连接到尾部
                while pre.next:
                    pre = pre.next
            length *= 2         
        return dummy.next
```

### 快速排序

* 不稳定排序，使用分治策略来把一个序列（list）分为较小和较大的2个子序列，然后递归地排序两个子序列。最好的情况时每次都能均匀划分序列，最坏情况是枢纽选为最大或最小的数字，退化为冒泡排序，时间复杂度达到O(n2)。

```python
import random
def quick_sort(low,high):
    if low < high:
        random_idx = random.randint(low,high)
        arr[low], arr[random_idx] = arr[random_idx], arr[low]
        cur, p = low, arr[low]
        for i in range(low+1,high+1):
            if arr[i] < p:
                cur += 1
                arr[cur], arr[i] = arr[i], arr[cur]
        arr[cur], arr[low] = arr[low], arr[cur]
        quick_sort(low,cur-1)
        quick_sort(cur+1,high)
        
quick_sort(0,len(arr)-1)
```

```python
# 简单写法
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
   	pivot = arr[-1]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left)+mid+quick_sort(right)
```

#### 215. 数组中第k个最大元素

* 只需要找到第 *k* 大的枢(pivot)即可，不需要对左右再进行排序。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partion(l,r):
            i = random.randint(l,r)
            nums[i], nums[l] = nums[l], nums[i]
            cur, p = l, nums[l]
            for j in range(l+1,r+1):
                if nums[j] < p:
                    cur += 1
                    nums[cur], nums[j] = nums[j], nums[cur]
            nums[cur], nums[l] = nums[l], nums[cur]
            if cur == t:
                return nums[cur]
            elif cur > t:
                return partion(l,cur-1)
            else:
                return partion(cur+1,r)

        n = len(nums)
        t = n - k
        return partion(0,n-1)
```

#### 75. 颜色分类

* 提前划分好三个区间，将遇到的0，2插入对应的区间。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        cur, white_start, white_end = 0, 0, len(nums)-1
        while cur <= white_end:
            if nums[cur]==0:
                nums[cur], nums[white_start] = nums[white_start], nums[cur]
                cur += 1
                white_start += 1
            elif nums[cur]==2:
                nums[cur], nums[white_end] = nums[white_end], nums[cur]
                white_end -= 1
            else:
                cur += 1
```

### 堆排序

* 利用堆这种数据结构所设计的一种不稳定排序算法，是对选择排序的优化。
  * 将无序序列构建成一个堆，根据升序降序需求选择大顶堆或小顶堆;
  * 将堆顶元素与末尾元素交换，将最大（小）元素"沉"到数组末端;
  * 重新调整结构，使其满足堆定义，然后继续交换堆顶元素与当前末尾元素，反复执行至序列有序。

```python
# 升序排序
def heap_sort(arr):
    n = len(arr)
    # 建堆，相当于在列表头部逐渐添加数据后调整结构
    for i in range(n-1,-1,-1):
        shift_down(n,i)
    #排序，将最大的元素移至末尾，类似于选择排序
    for i in range(n-1,0,-1):
        arr[0], arr[i] = arr[i], arr[0]
		shift_down(i,0)
        
def shift_down(n,i):
    j = (i<<1)+1
    while j < n:
        if j+1<n and arr[j+1]>arr[j]:
            j += 1
        if arr[j] > arr[i]:
            arr[i], arr[j] = arr[j], arr[i]
            i = j
            j = (i<<1)+1
        else:
            break
```

### 计数排序

* 作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。计数排序的基本思想是对于给定的输入序列中的每一个元素x，确定该序列中值小于x的元素的个数（此处并非比较各元素的大小，而是通过对元素值的计数和计数值的累加来确定）。一旦有了这个信息，就可以将x直接存放到最终的输出序列的正确位置上。相当于构建只存储单一键值的桶。

```python
def count_sort(arr):
    n = len(arr)
    if n < 2:
        return arr
    max_num = max(arr)
    buckets = [0] * (max_num+1)
    for each in arr:
        buckets[each] += 1
    res = []
    for i in range(max_num+1):
        res.extend(buckets[i]*[i])
    return res
```

### 桶排序

* 每个桶存储一定范围的数值，当要被排序的数组内的数值是均匀分配的时候，桶排序使用线性时间，若都分到了一个桶中则退化到nlogn。将数据放在几个有序的桶内（越多越好），将每个桶内的数据进行排序，最后有序地将每个桶中的数据从小到大依次取出，即完成了排序。

```python
def bucket_sort(arr):
    n = len(arr)
    min_num, max_num = min(arr), max(arr)
    bucket_range = (max_num-min_num)//n+1
    buckets = [[] for _ in range(n)]
    for each in arr:
        buckets[(each-min_num)//bucket_range].append(each)
    res = []
    for each in buckets:
        for x in sorted(each):
            res.append(x)
    return res
```

#### 451. 根据字符出现频率排序

* 先统计词频，然后存储从1到maxfreq每个词频出现的单词，按照降序遍历桶。

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        cnt = Counter(s)
        maxs = max(cnt.values())
        bucket = [[] for _ in range(maxs)]
        for k,v in cnt.items():
            bucket[v-1].append(k)
        res = []
        for i in range(maxs-1,-1,-1):
            for each in bucket[i]:
                res.append((i+1)*each)
        return ''.join(res)
```

#### 164. 最大间距

* 将差值最大的相邻元素分到不同的桶中，最大间距大于等于ceil((max-min)/(n-1))，取下界进行划分。

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        max_num, min_num = max(nums), min(nums)
        # 差值最大的相邻元素一定大于该值，被分到不同桶中
        bucket_range = max(1,(max_num-min_num)//(n-1))
        # 只用记录每个桶中的最大最小值
        buckets = [[-1]*2 for _ in range((max_num-min_num)//bucket_range+1)]  # 每个桶前闭后开
        '''
        也可以直接分成n个桶，最大相邻元素一定在两个桶中。
		bucket_range = (max_num-min_num)//n+1
		buckets = [[-1]*2 for _ in range(n)]
		'''
        for each in nums:
            idx = (each-min_num)//bucket_range
            if buckets[idx][0] == -1:
                buckets[idx][0] = buckets[idx][1] = each
            else:
                buckets[idx][0] = max(buckets[idx][0],each)
                buckets[idx][1] = min(buckets[idx][1],each)
        res, pre = 0, None
        for each in buckets:
            if each[0] == -1:
                continue
            if pre:     # 当前桶的最小值和前一个桶的最大值比较
                res = max(res, each[1]-pre)
            pre = each[0]
        return res
```



### 基数排序

* 根据键值的每位数字来分配桶。

```python
# 以10为基数，只考虑正整数
def radix_sort(arr):
    n = len(str(max(arr)))
    buckets = [[] for _ in range(10)]
    # 从最低位开始将数字分类
    for i in range(1,n+1):
        for each in arr:
            buckets[each%(10**i)//10**(i-1)].append(each)
        arr.clear()
        for each in buckets:
            arr.extend(each)
        bucket = [[] for _ in range(10)]
```





#### 

