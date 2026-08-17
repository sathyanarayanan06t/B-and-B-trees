"""
btree.py — B-Tree Implementation (Python Library)
Supports: insert, search, delete, and in-order traversal.

Terminology:
  t = minimum degree
  Every node holds at most (2t - 1) keys and at least (t - 1) keys (root may have fewer).
  All leaves sit at the same depth.
"""


class BTreeNode:
    """A single node in the B-Tree."""

    def __init__(self, leaf=True):
        self.keys = []
        self.children = []
        self.leaf = leaf


class BTree:
    """B-Tree with configurable minimum degree t."""

    def __init__(self, t=3):
        """Create an empty B-Tree. t must be >= 2."""
        self.root = BTreeNode(leaf=True)
        self.t = t

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def search(self, key, node=None):
        """Return True if key exists in the tree, False otherwise."""
        if node is None:
            node = self.root

        i = 0  # 'i' represents the index used for traversal
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and node.keys[i] == key:
            return True

        if node.leaf:
            return False

        return self.search(key, node.children[i])

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def insert(self, key):
        """Insert a key into the B-Tree."""
        root = self.root

        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root

        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        """Insert key into a node that is guaranteed to be non-full."""
        i = len(node.keys) - 1

        if node.leaf:
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        """Split parent.children[i] (which is full) into two nodes."""
        t = self.t  # 't' represents the minimum degree of the B-Tree
        full = parent.children[i]
        sibling = BTreeNode(leaf=full.leaf)

        parent.keys.insert(i, full.keys[t - 1])
        parent.children.insert(i + 1, sibling)

        sibling.keys = full.keys[t:]
        full.keys = full.keys[:t - 1]

        if not full.leaf:
            sibling.children = full.children[t:]
            full.children = full.children[:t]

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def delete(self, key):
        """Delete a key from the B-Tree (no-op if key absent)."""
        if not self.root.keys:
            return
        self._delete(self.root, key)

        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]

    def _delete(self, node, key):
        """Recursively remove key from the subtree rooted at node."""
        t = self.t  # 't' represents the minimum degree of the B-Tree
        i = 0  # 'i' represents the index used for traversal
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if node.leaf:
            if i < len(node.keys) and node.keys[i] == key:
                node.keys.pop(i)
            return

        if i < len(node.keys) and node.keys[i] == key:
            return self._delete_internal(node, i)

        if len(node.children[i].keys) < t:
            self._fill(node, i)

        if i > len(node.keys):
            i -= 1

        self._delete(node.children[i], key)

    def _delete_internal(self, node, i):
        """Handle deletion when node.keys[i] is inside an internal node."""
        t = self.t  # 't' represents the minimum degree of the B-Tree
        key = node.keys[i]

        if len(node.children[i].keys) >= t:
            pred = self._predecessor(node.children[i])
            node.keys[i] = pred
            self._delete(node.children[i], pred)

        elif len(node.children[i + 1].keys) >= t:
            succ = self._successor(node.children[i + 1])
            node.keys[i] = succ
            self._delete(node.children[i + 1], succ)

        else:
            self._merge(node, i)
            self._delete(node.children[i], key)

    def _predecessor(self, node):
        """Return the largest key in the subtree rooted at node."""
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1]

    def _successor(self, node):
        """Return the smallest key in the subtree rooted at node."""
        while not node.leaf:
            node = node.children[0]
        return node.keys[0]

    def _fill(self, node, i):
        """Ensure node.children[i] has at least t keys via borrow or merge."""
        t = self.t  # 't' represents the minimum degree of the B-Tree

        if i > 0 and len(node.children[i - 1].keys) >= t:
            self._borrow_prev(node, i)
        elif i < len(node.children) - 1 and len(node.children[i + 1].keys) >= t:
            self._borrow_next(node, i)
        else:
            if i < len(node.children) - 1:
                self._merge(node, i)
            else:
                self._merge(node, i - 1)

    def _borrow_prev(self, node, i):
        """Rotate a key from the left sibling through the parent."""
        child = node.children[i]
        left = node.children[i - 1]

        child.keys.insert(0, node.keys[i - 1])
        node.keys[i - 1] = left.keys.pop()

        if not child.leaf:
            child.children.insert(0, left.children.pop())

    def _borrow_next(self, node, i):
        """Rotate a key from the right sibling through the parent."""
        child = node.children[i]
        right = node.children[i + 1]

        child.keys.append(node.keys[i])
        node.keys[i] = right.keys.pop(0)

        if not child.leaf:
            child.children.append(right.children.pop(0))

    def _merge(self, node, i):
        """Merge children[i] and children[i+1] using keys[i] as separator."""
        left = node.children[i]
        right = node.children[i + 1]

        left.keys.append(node.keys.pop(i))
        left.keys.extend(right.keys)
        left.children.extend(right.children)

        node.children.pop(i + 1)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def traverse(self, node=None):
        """Return all keys in sorted order (in-order traversal)."""
        if node is None:
            node = self.root

        result = []
        for i in range(len(node.keys)):
            if not node.leaf:
                result.extend(self.traverse(node.children[i]))
            result.append(node.keys[i])

        if not node.leaf and node.children:
            result.extend(self.traverse(node.children[-1]))

        return result
