## 栈

栈是一种后进先出的线性数据结构。

#### 1003. 检查替换后的词是否有效

* 凑齐abc后将三个字符出栈，判断最后栈是否为空。

```python
 class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for char in s:
            if char == 'c' and stack[-2:] == ['a', 'b']:
                stack.pop()
                stack.pop()
            elif char in ['a', 'b']:
                stack.append(char)
        return not stack
```

