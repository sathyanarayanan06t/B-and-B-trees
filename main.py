"""main.py - Demonstration of btree_bplustree_lib (B-Tree and B+ Tree). Run: python main.py
Every test prints the tree and compares the result with the answer worked out by hand."""

import random

from btree_bplustree_lib import BTree, BTreeNode, BPlusTree

results = {"passed": 0, "failed": 0}

def check(description, actual, expected):
    """Compare actual with expected and print PASS / FAIL."""
    if actual == expected:
        results["passed"] += 1
        print(f"  [PASS] {description}")
    else:
        results["failed"] += 1
        print(f"  [FAIL] {description}\n         expected: {expected}\n         actual  : {actual}")

def L(*groups):
    """Shortcut for one expected level of letter nodes: L("AB", "CD") -> [['A','B'], ['C','D']]"""
    return [list(g) for g in groups]

def make_node(keys, children=None):
    """Build a B-Tree node by hand (used to set up a known starting tree)."""
    node = BTreeNode(leaf=(children is None))
    node.keys, node.children = list(keys), children or []
    return node

def step(tree, message, expected_levels):
    """Print what happened, show the tree, and check its shape level by level."""
    print("\n  " + message)
    tree.print_tree()
    check(message.split(" (")[0], tree.levels(), expected_levels)

def random_test(tree):
    """1000 random inserts + 500 deletes, compared with a Python set (the expected answer)."""
    keys = random.sample(range(10000), 1000)
    reference = set(keys)
    for k in keys:
        tree.insert(k)
    check("inorder == sorted(inserted keys)", tree.inorder(), sorted(reference))
    to_delete = random.sample(keys, 500)
    for k in to_delete:
        tree.delete(k)
    reference -= set(to_delete)
    check("inorder == sorted(remaining keys)", tree.inorder(), sorted(reference))
    check("deleted keys are not found", any(tree.search(k) for k in to_delete), False)
    check("remaining keys are found", all(tree.search(k) for k in reference), True)

def test_btree():
    print("=" * 60 + "\nB-TREE\n" + "=" * 60)

    print("\nTest 1: insert keys 1..10 with t = 2")
    tree = BTree(t=2)
    for k in range(1, 11):
        tree.insert(k)
    tree.print_tree()
    check("inorder",   tree.inorder(),   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    check("preorder",  tree.preorder(),  [4, 2, 1, 3, 6, 8, 5, 7, 9, 10])
    check("postorder", tree.postorder(), [1, 3, 2, 5, 7, 9, 10, 6, 8, 4])
    check("height", tree.height(), 2)
    check("search 7 (present)", tree.search(7), True)
    check("search 11 (absent)", tree.search(11), False)
    tree.insert(5)
    check("inserting duplicate 5 changes nothing", tree.inorder(), list(range(1, 11)))

    print("\nTest 2: insertions with t = 3 (letters as keys). Starting tree:")
    tree = BTree(t=3)
    tree.root = make_node("GMPX", [make_node("ACDE"), make_node("JK"), make_node("NO"),
                                   make_node("RSTUV"), make_node("YZ")])
    tree.print_tree()
    tree.insert("B")
    step(tree, "Insert B (leaf is not full)", [L("GMPX"), L("ABCDE", "JK", "NO", "RSTUV", "YZ")])
    tree.insert("Q")
    step(tree, "Insert Q (leaf RSTUV full -> split, T moves up)",
         [L("GMPTX"), L("ABCDE", "JK", "NO", "QRS", "UV", "YZ")])
    tree.insert("L")
    step(tree, "Insert L (root full -> split, tree grows taller)",
         [L("P"), L("GM", "TX"), L("ABCDE", "JKL", "NO", "QRS", "UV", "YZ")])
    tree.insert("F")
    step(tree, "Insert F (leaf ABCDE full -> split, C moves up)",
         [L("P"), L("CGM", "TX"), L("AB", "DEF", "JKL", "NO", "QRS", "UV", "YZ")])

    print("\nTest 3: deletions with t = 3 (continuing from Test 2)")
    tree.delete("F")
    step(tree, "Delete F (case 1: key in a leaf)",
         [L("P"), L("CGM", "TX"), L("AB", "DE", "JKL", "NO", "QRS", "UV", "YZ")])
    tree.delete("M")
    step(tree, "Delete M (case 2a: replace by predecessor L)",
         [L("P"), L("CGL", "TX"), L("AB", "DE", "JK", "NO", "QRS", "UV", "YZ")])
    tree.delete("G")
    step(tree, "Delete G (case 2c: merge DE + G + JK)",
         [L("P"), L("CL", "TX"), L("AB", "DEJK", "NO", "QRS", "UV", "YZ")])
    tree.delete("D")
    step(tree, "Delete D (case 3b: merge, root shrinks)",
         [L("CLPTX"), L("AB", "EJK", "NO", "QRS", "UV", "YZ")])
    tree.delete("B")
    step(tree, "Delete B (case 3a: borrow from sibling)",
         [L("ELPTX"), L("AC", "JK", "NO", "QRS", "UV", "YZ")])
    check("delete of a missing key returns False", tree.delete("Z2"), False)

    print("\nTest 4: random test (t = 3)")
    random_test(BTree(t=3))

def test_bplus():
    print("\n" + "=" * 60 + "\nB+ TREE\n" + "=" * 60)

    print("\nTest 1: insert keys 1..10 with order m = 4 (max 3 keys per node)")
    tree = BPlusTree(m=4)
    for k in range(1, 11):
        tree.insert(k)
    tree.print_tree()
    check("inorder (leaf chain)", tree.inorder(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    check("preorder",  tree.preorder(),  [7, 3, 5, 1, 2, 3, 4, 5, 6, 9, 7, 8, 9, 10])
    check("postorder", tree.postorder(), [1, 2, 3, 4, 5, 6, 3, 5, 7, 8, 9, 10, 9, 7])
    check("height", tree.height(), 2)
    check("search 7 (present)", tree.search(7), True)
    check("search 11 (absent)", tree.search(11), False)
    check("range_query(4, 8)", tree.range_query(4, 8), [4, 5, 6, 7, 8])
    check("range_query(11, 20) (empty)", tree.range_query(11, 20), [])

    print("\nTest 2: deletions on the tree from Test 1")
    check("delete of a missing key returns False", tree.delete(99), False)
    tree.delete(7)
    step(tree, "Delete 7 (merge leaves, parent merges, root shrinks)",
         [[[3, 5, 7]], [[1, 2], [3, 4], [5, 6], [8, 9, 10]]])
    print("  (7 stays as a separator in the root - it only guides the search)")
    check("search 7 after delete", tree.search(7), False)
    tree.delete(9)
    step(tree, "Delete 9 (leaf still has enough keys)", [[[3, 5, 7]], [[1, 2], [3, 4], [5, 6], [8, 10]]])
    tree.delete(6)
    step(tree, "Delete 6 (merge with left sibling)", [[[3, 7]], [[1, 2], [3, 4, 5], [8, 10]]])
    tree.delete(1)
    step(tree, "Delete 1 (borrow from right sibling)", [[[4, 7]], [[2, 3], [4, 5], [8, 10]]])
    check("range_query(3, 9) after deletions", tree.range_query(3, 9), [3, 4, 5, 8])

    print("\nTest 3: random test (m = 4)")
    random_test(BPlusTree(m=4))

def main():
    random.seed(42)          # fixed seed so every run gives the same random tests
    test_btree()
    test_bplus()
    print(f"\nTEST SUMMARY: {results['passed']} passed, {results['failed']} failed")

if __name__ == "__main__":
    main()
