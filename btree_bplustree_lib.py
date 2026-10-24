"""
btree_bplustree_lib.py - A small Python library for B-Trees and B+ Trees.

B-Tree
    Follows CLRS (Introduction to Algorithms, 4th ed.), Chapter 18.
    Function names match the textbook procedures:
        B-TREE-SEARCH          -> BTree.search / BTree._search
        B-TREE-SPLIT-CHILD     -> BTree._split_child
        B-TREE-INSERT          -> BTree.insert
        B-TREE-INSERT-NONFULL  -> BTree._insert_nonfull
        B-TREE-DELETE          -> BTree.delete / BTree._delete  (cases 1, 2a-2c, 3a-3b)

B+ Tree
    Follows the B+ tree algorithms in Silberschatz, Korth & Sudarshan,
    "Database System Concepts" (chapter "Indexing"):
        find            -> BPlusTree.search / BPlusTree._find_leaf
        insert          -> BPlusTree.insert
        insert_in_parent-> BPlusTree._insert_in_parent
        delete_entry    -> BPlusTree._delete_entry
    plus range_query, which walks the linked list of leaves.

Both trees store unique keys only (inserting a key that already exists does nothing).
Keys can be anything that can be compared with < and == (ints, strings, ...).
"""


# =============================================================================
#                 Printing helpers (shared by both trees)
# =============================================================================

def node_text(node):
    """A node's keys as text, e.g. [3 5 7]"""
    return "[" + " ".join(str(k) for k in node.keys) + "]"


def print_children(node, indent):
    """
    Print every child of `node` on its own line, then (recursively) its children.
      * "|-- " is drawn in front of a child that has more siblings below it,
      * "`-- " is drawn in front of the last child.
    `indent` is the text drawn in front of this level: under a child that has
    more siblings we continue the line with "|   ", otherwise just spaces.
    """
    for i, child in enumerate(node.children):
        is_last = (i == len(node.children) - 1)
        if is_last:
            print(indent + "`-- " + node_text(child))
            print_children(child, indent + "    ")
        else:
            print(indent + "|-- " + node_text(child))
            print_children(child, indent + "|   ")


# =============================================================================
#                                   B-TREE
# =============================================================================

class BTreeNode:
    """One node of a B-Tree."""

    def __init__(self, leaf=True):
        self.keys = []        # keys in sorted order   (CLRS: x.key1 .. x.keyn)
        self.children = []    # child pointers         (CLRS: x.c1 .. x.c(n+1))
        self.leaf = leaf      # True if node has no children (CLRS: x.leaf)


class BTree:
    """
    B-Tree with minimum degree t (CLRS definition):
      * every node except the root has at least t-1 keys
      * every node has at most 2t-1 keys
      * all leaves are at the same depth
    """

    def __init__(self, t=2):
        if t < 2:
            raise ValueError("minimum degree t must be at least 2")
        self.t = t
        self.root = BTreeNode(leaf=True)        # B-TREE-CREATE: an empty leaf

    # ------------------------------------------------------------------ search
    def search(self, k):
        """Return True if key k is in the tree."""
        return self._search(self.root, k) is not None

    def _search(self, x, k):
        """B-TREE-SEARCH: returns (node, index) if k is found, else None."""
        i = 0
        while i < len(x.keys) and k > x.keys[i]:     # find first key >= k
            i += 1
        if i < len(x.keys) and k == x.keys[i]:       # found it
            return (x, i)
        if x.leaf:                                   # nowhere else to look
            return None
        return self._search(x.children[i], k)        # go down to child i

    # ------------------------------------------------------------------ insert
    def insert(self, k):
        """B-TREE-INSERT: insert key k (duplicates are ignored)."""
        if self.search(k):
            return
        r = self.root
        if len(r.keys) == 2 * self.t - 1:            # root is full -> tree grows taller
            s = BTreeNode(leaf=False)
            s.children.append(r)
            self.root = s
            self._split_child(s, 0)
            self._insert_nonfull(s, k)
        else:
            self._insert_nonfull(r, k)

    def _split_child(self, x, i):
        """
        B-TREE-SPLIT-CHILD: child y = x.children[i] is full (2t-1 keys).
        Split it into y (first t-1 keys) and z (last t-1 keys) and
        move the middle key up into x.
        """
        t = self.t
        y = x.children[i]
        z = BTreeNode(leaf=y.leaf)

        middle_key = y.keys[t - 1]
        z.keys = y.keys[t:]              # last t-1 keys go to z
        y.keys = y.keys[:t - 1]          # first t-1 keys stay in y
        if not y.leaf:
            z.children = y.children[t:]  # last t children go to z
            y.children = y.children[:t]  # first t children stay in y

        x.children.insert(i + 1, z)      # z becomes the child right after y
        x.keys.insert(i, middle_key)     # middle key moves up into x

    def _insert_nonfull(self, x, k):
        """B-TREE-INSERT-NONFULL: insert k into the subtree rooted at non-full node x."""
        i = len(x.keys) - 1
        if x.leaf:
            while i >= 0 and k < x.keys[i]:          # find position for k
                i -= 1
            x.keys.insert(i + 1, k)
        else:
            while i >= 0 and k < x.keys[i]:          # find the child to go down to
                i -= 1
            i += 1
            if len(x.children[i].keys) == 2 * self.t - 1:   # child full -> split first
                self._split_child(x, i)
                if k > x.keys[i]:                    # which half does k belong in?
                    i += 1
            self._insert_nonfull(x.children[i], k)

    # ------------------------------------------------------------------ delete
    def delete(self, k):
        """B-TREE-DELETE: remove key k. Returns True if it was found."""
        found = self._delete(self.root, k)
        # If the root lost its last key, its only child becomes the new root
        # (this is the only way a B-Tree gets shorter).
        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]
        return found

    def _delete(self, x, k):
        """
        Delete k from the subtree rooted at x.
        Invariant (CLRS): whenever we call _delete on a node x that is not the
        root, x has at least t keys, so we can remove one key without breaking
        the B-Tree rules.
        """
        t = self.t
        i = 0
        while i < len(x.keys) and k > x.keys[i]:
            i += 1

        # ---- k is in this node ----
        if i < len(x.keys) and x.keys[i] == k:
            if x.leaf:
                # Case 1: k is in a leaf -> just remove it.
                x.keys.pop(i)
                return True

            y = x.children[i]          # child before k
            z = x.children[i + 1]      # child after k
            if len(y.keys) >= t:
                # Case 2a: y has >= t keys -> replace k by its predecessor.
                pred = self._max_key(y)
                x.keys[i] = pred
                return self._delete(y, pred)
            elif len(z.keys) >= t:
                # Case 2b: z has >= t keys -> replace k by its successor.
                succ = self._min_key(z)
                x.keys[i] = succ
                return self._delete(z, succ)
            else:
                # Case 2c: both have t-1 keys -> merge k and z into y, then delete from y.
                self._merge(x, i)
                return self._delete(y, k)

        # ---- k is not in this node ----
        if x.leaf:
            return False               # k is not in the tree

        # Case 3: k (if present) is in subtree children[i].
        # Make sure that child has at least t keys before going down.
        if len(x.children[i].keys) == t - 1:
            i = self._fill_child(x, i)
        return self._delete(x.children[i], k)

    def _fill_child(self, x, i):
        """
        Child x.children[i] has only t-1 keys. Give it one more key
        (case 3a: borrow from a sibling, case 3b: merge with a sibling).
        Returns the index of the child that now holds the keys of the old child i.
        """
        t = self.t
        child = x.children[i]

        # Case 3a (left): left sibling has >= t keys -> rotate a key right.
        if i > 0 and len(x.children[i - 1].keys) >= t:
            left = x.children[i - 1]
            child.keys.insert(0, x.keys[i - 1])      # parent key comes down
            x.keys[i - 1] = left.keys.pop()          # sibling's last key goes up
            if not left.leaf:
                child.children.insert(0, left.children.pop())
            return i

        # Case 3a (right): right sibling has >= t keys -> rotate a key left.
        if i < len(x.keys) and len(x.children[i + 1].keys) >= t:
            right = x.children[i + 1]
            child.keys.append(x.keys[i])             # parent key comes down
            x.keys[i] = right.keys.pop(0)            # sibling's first key goes up
            if not right.leaf:
                child.children.append(right.children.pop(0))
            return i

        # Case 3b: both siblings have t-1 keys -> merge with one of them.
        if i < len(x.keys):
            self._merge(x, i)          # merge child i with right sibling
            return i
        else:
            self._merge(x, i - 1)      # last child: merge with left sibling
            return i - 1

    def _merge(self, x, i):
        """Merge x.children[i], x.keys[i] and x.children[i+1] into x.children[i]."""
        y = x.children[i]
        z = x.children[i + 1]
        y.keys.append(x.keys.pop(i))   # separator key comes down from x
        y.keys.extend(z.keys)
        y.children.extend(z.children)
        x.children.pop(i + 1)          # z is no longer needed

    def _max_key(self, x):
        """Largest key in subtree x (keep going right)."""
        while not x.leaf:
            x = x.children[-1]
        return x.keys[-1]

    def _min_key(self, x):
        """Smallest key in subtree x (keep going left)."""
        while not x.leaf:
            x = x.children[0]
        return x.keys[0]

    # -------------------------------------------------------------- traversals
    def inorder(self):
        """Keys in sorted order: child0, key0, child1, key1, ..., child_n."""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, x, result):
        for i in range(len(x.keys)):
            if not x.leaf:
                self._inorder(x.children[i], result)
            result.append(x.keys[i])
        if not x.leaf:
            self._inorder(x.children[-1], result)

    def preorder(self):
        """Visit a node's keys first, then its children from left to right."""
        result = []
        self._preorder(self.root, result)
        return result

    def _preorder(self, x, result):
        result.extend(x.keys)
        for child in x.children:
            self._preorder(child, result)

    def postorder(self):
        """Visit a node's children from left to right first, then its keys."""
        result = []
        self._postorder(self.root, result)
        return result

    def _postorder(self, x, result):
        for child in x.children:
            self._postorder(child, result)
        result.extend(x.keys)

    # ------------------------------------------------------------ print / info
    def levels(self):
        """Level-order view: a list of levels, each level a list of nodes' key lists."""
        result = []
        current = [self.root]
        while current:
            result.append([node.keys[:] for node in current])
            next_level = []
            for node in current:
                next_level.extend(node.children)
            current = next_level
        return result

    def print_tree(self):
        """
        Print the tree sideways, one node per line, e.g.
            [4]
            |-- [2]
            |   |-- [1]
            |   `-- [3]
            `-- [6 8]
                |-- [5]
                ...
        """
        print(node_text(self.root))
        print_children(self.root, "")

    def height(self):
        """Number of edges from the root to a leaf (a single node has height 0)."""
        h = 0
        x = self.root
        while not x.leaf:
            x = x.children[0]
            h += 1
        return h


# =============================================================================
#                                  B+ TREE
# =============================================================================

class BPlusNode:
    """One node of a B+ Tree."""

    def __init__(self, leaf=True):
        self.keys = []        # sorted keys
        self.children = []    # child pointers (internal nodes only)
        self.leaf = leaf
        self.next = None      # leaves only: pointer to the next leaf on the right


class BPlusTree:
    """
    B+ Tree of order m (m = maximum number of children of a node):
      * every node has at most m-1 keys
      * internal nodes (except root) have at least ceil(m/2) children
      * leaves (except root) have at least ceil((m-1)/2) keys
      * ALL keys are stored in the leaves; internal nodes only hold copies
        used for routing. The leaves are linked left to right.
    In a node, child[i] holds keys  < keys[i]  and child[i+1] holds keys >= keys[i].
    """

    def __init__(self, m=4):
        if m < 3:
            raise ValueError("order m must be at least 3")
        self.m = m
        self.max_keys = m - 1
        self.min_leaf_keys = m // 2                 # = ceil((m-1)/2)
        self.min_children = (m + 1) // 2            # = ceil(m/2)
        self.root = BPlusNode(leaf=True)

    # ------------------------------------------------------------------ search
    def _find_leaf(self, k):
        """
        Walk down from the root to the leaf where k is (or should be).
        Returns (leaf, path) where path is the list of internal nodes visited,
        so path[-1] is the leaf's parent.
        """
        node = self.root
        path = []
        while not node.leaf:
            path.append(node)
            i = 0
            while i < len(node.keys) and k >= node.keys[i]:
                i += 1
            node = node.children[i]
        return node, path

    def search(self, k):
        """Return True if key k is in the tree."""
        leaf, _ = self._find_leaf(k)
        return k in leaf.keys

    def range_query(self, low, high):
        """All keys with low <= key <= high, found by walking the leaf chain."""
        result = []
        leaf, _ = self._find_leaf(low)
        while leaf is not None:
            for k in leaf.keys:
                if k > high:
                    return result
                if k >= low:
                    result.append(k)
            leaf = leaf.next
        return result

    # ------------------------------------------------------------------ insert
    def insert(self, k):
        """Insert key k (duplicates are ignored)."""
        leaf, path = self._find_leaf(k)
        if k in leaf.keys:
            return

        # insert_in_leaf: put k at its sorted position
        i = 0
        while i < len(leaf.keys) and leaf.keys[i] < k:
            i += 1
        leaf.keys.insert(i, k)

        if len(leaf.keys) > self.max_keys:
            # Leaf overflow: split into two leaves. The first ceil(m/2) keys stay.
            split = (self.m + 1) // 2
            new_leaf = BPlusNode(leaf=True)
            new_leaf.keys = leaf.keys[split:]
            leaf.keys = leaf.keys[:split]

            new_leaf.next = leaf.next        # keep the leaf chain linked
            leaf.next = new_leaf

            # The smallest key of the new leaf is COPIED up to the parent.
            self._insert_in_parent(leaf, new_leaf.keys[0], new_leaf, path)

    def _insert_in_parent(self, left, k, right, path):
        """Add separator key k and new node `right` (just after `left`) to left's parent."""
        if left is self.root:
            # The root was split -> make a new root. Tree grows taller.
            new_root = BPlusNode(leaf=False)
            new_root.keys = [k]
            new_root.children = [left, right]
            self.root = new_root
            return

        parent = path.pop()
        i = parent.children.index(left)
        parent.keys.insert(i, k)
        parent.children.insert(i + 1, right)

        if len(parent.children) > self.m:
            # Internal overflow: split the parent. The middle key is MOVED up
            # (unlike a leaf split, where it is copied).
            split = (self.m + 2) // 2                 # = ceil((m+1)/2) children stay
            new_node = BPlusNode(leaf=False)
            middle_key = parent.keys[split - 1]

            new_node.keys = parent.keys[split:]
            new_node.children = parent.children[split:]
            parent.keys = parent.keys[:split - 1]
            parent.children = parent.children[:split]

            self._insert_in_parent(parent, middle_key, new_node, path)

    # ------------------------------------------------------------------ delete
    def delete(self, k):
        """Remove key k. Returns True if it was found."""
        leaf, path = self._find_leaf(k)
        if k not in leaf.keys:
            return False
        leaf.keys.remove(k)
        # Note: a copy of k may still sit in an internal node as a separator.
        # That is fine - separators only guide the search, they are not data.
        self._delete_entry(leaf, path)
        return True

    def _too_few(self, node):
        """True if a non-root node has fallen below the minimum size."""
        if node.leaf:
            return len(node.keys) < self.min_leaf_keys
        return len(node.children) < self.min_children

    def _delete_entry(self, node, path):
        """
        Called after something was removed from `node`. Fix underflow by
        merging (coalescing) with a sibling or borrowing (redistributing) from it.
        """
        if node is self.root:
            # Root with a single child: that child becomes the root. Tree gets shorter.
            if not node.leaf and len(node.children) == 1:
                self.root = node.children[0]
            return

        if not self._too_few(node):
            return

        parent = path[-1]
        i = parent.children.index(node)

        # Pick a sibling: the left one if it exists, otherwise the right one.
        if i > 0:
            left, right = parent.children[i - 1], node
            sep = i - 1                          # index of key between left and right
        else:
            left, right = node, parent.children[1]
            sep = 0
        sibling = left if right is node else right

        # Can both nodes fit into a single node?
        if node.leaf:
            fits = len(left.keys) + len(right.keys) <= self.max_keys
        else:
            fits = len(left.children) + len(right.children) <= self.m

        if fits:
            # ---- Coalesce: move everything from `right` into `left` ----
            if node.leaf:
                left.keys.extend(right.keys)
                left.next = right.next           # unlink `right` from the leaf chain
            else:
                left.keys.append(parent.keys[sep])   # separator comes down
                left.keys.extend(right.keys)
                left.children.extend(right.children)
            # Remove the separator and the pointer to `right` from the parent,
            # which may make the parent underflow in turn.
            parent.keys.pop(sep)
            parent.children.pop(sep + 1)
            self._delete_entry(parent, path[:-1])
        else:
            # ---- Redistribute: borrow one entry from the sibling ----
            if sibling is left:                  # borrow the LAST entry of left sibling
                if node.leaf:
                    node.keys.insert(0, left.keys.pop())
                    parent.keys[sep] = node.keys[0]
                else:
                    node.keys.insert(0, parent.keys[sep])
                    node.children.insert(0, left.children.pop())
                    parent.keys[sep] = left.keys.pop()
            else:                                # borrow the FIRST entry of right sibling
                if node.leaf:
                    node.keys.append(right.keys.pop(0))
                    parent.keys[sep] = right.keys[0]
                else:
                    node.keys.append(parent.keys[sep])
                    node.children.append(right.children.pop(0))
                    parent.keys[sep] = right.keys.pop(0)

    # -------------------------------------------------------------- traversals
    def inorder(self):
        """All data keys in sorted order: start at the leftmost leaf, follow `next`."""
        node = self.root
        while not node.leaf:
            node = node.children[0]
        result = []
        while node is not None:
            result.extend(node.keys)
            node = node.next
        return result

    def preorder(self):
        """Visit a node's keys first, then its children (includes routing keys)."""
        result = []
        self._preorder(self.root, result)
        return result

    def _preorder(self, node, result):
        result.extend(node.keys)
        for child in node.children:
            self._preorder(child, result)

    def postorder(self):
        """Visit a node's children first, then its keys (includes routing keys)."""
        result = []
        self._postorder(self.root, result)
        return result

    def _postorder(self, node, result):
        for child in node.children:
            self._postorder(child, result)
        result.extend(node.keys)

    # ------------------------------------------------------------ print / info
    def levels(self):
        """Level-order view: a list of levels, each level a list of nodes' key lists."""
        result = []
        current = [self.root]
        while current:
            result.append([node.keys[:] for node in current])
            next_level = []
            for node in current:
                next_level.extend(node.children)
            current = next_level
        return result

    def print_tree(self):
        """Print the tree sideways, one node per line (same format as BTree.print_tree)."""
        print(node_text(self.root))
        print_children(self.root, "")

    def height(self):
        """Number of edges from the root to a leaf (a single node has height 0)."""
        h = 0
        node = self.root
        while not node.leaf:
            node = node.children[0]
            h += 1
        return h
