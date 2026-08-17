"""
bplus_tree.py — B+ Tree Implementation (Python Library)
Supports: insert, search, delete, range_query, and leaf-level traversal.

Key differences from a B-Tree:
  1. All data lives in the LEAF nodes; internal nodes store routing copies.
  2. Leaf nodes are linked left-to-right for efficient range queries.
  3. Search always descends to a leaf.

Terminology:
  order = maximum number of children per internal node.
  Each node holds at most (order - 1) keys.
"""


class _LeafNode:
    """Leaf node — holds actual keys and a next-pointer for the linked list."""

    def __init__(self):
        self.keys = []
        self.next = None


class _InternalNode:
    """Internal node — holds routing keys and child pointers."""

    def __init__(self):
        self.keys = []
        self.children = []


class BPlusTree:
    """B+ Tree with configurable order."""

    def __init__(self, order=5):
        """Create an empty B+ Tree. order must be >= 3."""
        self.root = _LeafNode()
        self.order = order

    @property
    def _max_keys(self):
        """Max keys any node can hold before a split is required."""
        return self.order - 1

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def search(self, key):
        """Return True if key exists in the tree, False otherwise."""
        leaf = self._find_leaf(key)
        return key in leaf.keys

    def _find_leaf(self, key):
        """Navigate from the root down to the leaf that should contain key."""
        node = self.root
        while isinstance(node, _InternalNode):
            i = 0  # 'i' represents the index used for traversal
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def insert(self, key):
        """Insert a key into the B+ Tree."""
        path = []
        node = self.root

        while isinstance(node, _InternalNode):
            path.append(node)
            i = 0  # 'i' represents the index used for traversal
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]

        leaf = node
        i = 0  # 'i' represents the index used for traversal
        while i < len(leaf.keys) and key > leaf.keys[i]:
            i += 1
        leaf.keys.insert(i, key)

        if len(leaf.keys) > self._max_keys:
            new_leaf = _LeafNode()
            mid = len(leaf.keys) // 2

            new_leaf.keys = leaf.keys[mid:]
            leaf.keys = leaf.keys[:mid]

            new_leaf.next = leaf.next
            leaf.next = new_leaf

            self._push_up(path, leaf, new_leaf.keys[0], new_leaf)

    def _push_up(self, path, left, key, right):
        """Promote key into the parent; split the parent if it overflows."""
        if not path:
            new_root = _InternalNode()
            new_root.keys = [key]
            new_root.children = [left, right]
            self.root = new_root
            return

        parent = path.pop()

        i = 0  # 'i' represents the index used for traversal
        while i < len(parent.keys) and key > parent.keys[i]:
            i += 1
        parent.keys.insert(i, key)
        parent.children.insert(i + 1, right)

        if len(parent.keys) > self._max_keys:
            mid = len(parent.keys) // 2
            promote = parent.keys[mid]

            new_internal = _InternalNode()
            new_internal.keys = parent.keys[mid + 1:]
            new_internal.children = parent.children[mid + 1:]

            parent.keys = parent.keys[:mid]
            parent.children = parent.children[:mid + 1]

            self._push_up(path, parent, promote, new_internal)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def range_query(self, low, high):
        """Return all keys k where low <= k <= high, in sorted order."""
        leaf = self._find_leaf(low)
        result = []

        while leaf is not None:
            for k in leaf.keys:  # 'k' represents the current key being processed
                if k > high:
                    return result
                if k >= low:
                    result.append(k)
            leaf = leaf.next

        return result

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def delete(self, key):
        """Delete a key from the B+ Tree (no-op if absent)."""
        path = []
        node = self.root

        while isinstance(node, _InternalNode):
            i = 0  # 'i' represents the index used for traversal
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            path.append((node, i))
            node = node.children[i]

        if key not in node.keys:
            return

        node.keys.remove(key)

        if not path:
            return

        min_keys = (self.order - 1) // 2
        if len(node.keys) >= min_keys:
            return

        self._fix_leaf(path, node)

    def _fix_leaf(self, path, leaf):
        """Repair an underflowing leaf by borrowing or merging."""
        min_keys = (self.order - 1) // 2
        parent, idx = path[-1]

        if idx > 0:
            left = parent.children[idx - 1]
            if len(left.keys) > min_keys:
                leaf.keys.insert(0, left.keys.pop())
                parent.keys[idx - 1] = leaf.keys[0]
                return

        if idx < len(parent.children) - 1:
            right = parent.children[idx + 1]
            if len(right.keys) > min_keys:
                leaf.keys.append(right.keys.pop(0))
                parent.keys[idx] = right.keys[0]
                return

        if idx > 0:
            left = parent.children[idx - 1]
            left.keys.extend(leaf.keys)
            left.next = leaf.next
            parent.keys.pop(idx - 1)
            parent.children.pop(idx)
        else:
            right = parent.children[idx + 1]
            leaf.keys.extend(right.keys)
            leaf.next = right.next
            parent.keys.pop(idx)
            parent.children.pop(idx + 1)

        self._fix_internal(path[:-1], parent)

    def _fix_internal(self, path, node):
        """Repair an underflowing internal node after a child merge."""
        if not path:
            if len(node.keys) == 0 and node.children:
                self.root = node.children[0]
            return

        min_keys = (self.order - 1) // 2
        if len(node.keys) >= min_keys:
            return

        parent, idx = path[-1]

        if idx > 0:
            left = parent.children[idx - 1]
            if len(left.keys) > min_keys:
                node.keys.insert(0, parent.keys[idx - 1])
                node.children.insert(0, left.children.pop())
                parent.keys[idx - 1] = left.keys.pop()
                return

        if idx < len(parent.children) - 1:
            right = parent.children[idx + 1]
            if len(right.keys) > min_keys:
                node.keys.append(parent.keys[idx])
                node.children.append(right.children.pop(0))
                parent.keys[idx] = right.keys.pop(0)
                return

        if idx > 0:
            left = parent.children[idx - 1]
            left.keys.append(parent.keys.pop(idx - 1))
            left.keys.extend(node.keys)
            left.children.extend(node.children)
            parent.children.pop(idx)
        else:
            right = parent.children[idx + 1]
            node.keys.append(parent.keys.pop(idx))
            node.keys.extend(right.keys)
            node.children.extend(right.children)
            parent.children.pop(idx + 1)

        self._fix_internal(path[:-1], parent)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    def traverse(self):
        """Return all keys in sorted order using the leaf linked list."""
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[0]

        result = []
        while node is not None:
            result.extend(node.keys)
            node = node.next
        return result
