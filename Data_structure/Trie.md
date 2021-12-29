## 字典树

Trie是一种用于实现字符串快速检索的多叉树结构。其每个节点具有若干个字符指针，若在插入或检索字符串时扫描到一个字符c，就沿着当前节点的c字符指针，走向该指针指向的节点。

#### 构建字典树

```python
# 利用dict
class Trie:
    def __init__(self):
        self.lookup = {}

    def insert(self, word):
        tree = self.lookup
        for a in word:
            if a not in tree:
                tree[a] = {}
            tree = tree[a]
        tree['#'] = '#'    

    def search(self, word):
        tree = self.lookup
        for a in word:
            if a not in tree:
                return False
            tree = tree[a]
        return True if '#' in tree else False

    def startsWith(self, prefix):
        tree = self.lookup
        for a in prefix:
            if a not in tree:
                return False
            tree = tree[a]
        return True

# 利用自建类
class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def search(self, word):
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node is not None and node.isEnd
```

#### 472.连接词

* 将字符串按长度排序，检查当前字符串能否由已插入的字符串构成，将不能构成的字符串插入字典树。

```python
class Trie:
    def __init__(self):
        self.children = [None]*26
        self.isEnd = False
    
    def insert(self,word):
        cur = self
        for each in word:
            idx = ord(each)-ord('a')
            if not cur.children[idx]:
                cur.children[idx] = Trie()
            cur = cur.children[idx]
        cur.isEnd = True
    
    def search(self,word,cnt):
        if not word and cnt>=2:
            return True
        cur = self
        for i,each in enumerate(word):
            cur = cur.children[ord(each)-ord('a')]
            if not cur:
                return False
            if cur.isEnd and self.search(word[i+1:],cnt+1):
                return True        
        return False


class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=lambda x: len(x))    # 长词由短词构成
        trie = Trie()
        res = []
        for i,each in enumerate(words):
            if not each:
                continue
            if trie.search(each,0):
                res.append(each)
            else:           # 连接词不需要加入树中
                trie.insert(each)
        return res
```

