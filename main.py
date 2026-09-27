"""
main.py - Demonstration of btree_lib (B-Tree and B+ Tree).

Part 1: Correctness tests on small trees whose results can be checked by hand.
Part 2: Runtime measurements for different (large) input sizes.
Part 3: Plots of the measurements (saved to runtime_plots.png).

Run:  python main.py        (needs matplotlib:  pip install matplotlib)
"""

import gc
import random
import time

import matplotlib.pyplot as plt

from btree_lib import BTree, BTreeNode, BPlusTree


# =============================================================================
#                         Small helpers for the tests
# =============================================================================

results = {"passed": 0, "failed": 0}


def check(description, actual, expected):
    """Compare actual with expected and print PASS / FAIL."""
    if actual == expected:
        results["passed"] += 1
        print(f"  [PASS] {description}")
    else:
        results["failed"] += 1
        print(f"  [FAIL] {description}")
        print(f"         expected: {expected}")
        print(f"         actual  : {actual}")


def L(*groups):
    """Shortcut for writing one expected level of letter nodes: L("AB", "CD") -> [['A','B'], ['C','D']]"""
    return [list(g) for g in groups]


def make_node(keys, children=None):
    """Build a B-Tree node by hand (used to set up a known starting tree)."""
    node = BTreeNode(leaf=(children is None))
    node.keys = list(keys)
    node.children = children if children is not None else []
    return node


def heading(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
#                        PART 1a: B-Tree correctness
# =============================================================================

def test_btree():
    heading("PART 1a: B-TREE CORRECTNESS TESTS")

    # ---------------- Test 1: insert 1..10 with t = 2 ----------------
    print("\nTest 1: insert keys 1..10 into an empty B-Tree with t = 2")
    tree = BTree(t=2)
    for k in range(1, 11):
        tree.insert(k)
    tree.print_tree()
    check("tree shape",   tree.levels(),
          [[[4]], [[2], [6, 8]], [[1], [3], [5], [7], [9, 10]]])
    check("inorder",   tree.inorder(),   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    check("preorder",  tree.preorder(),  [4, 2, 1, 3, 6, 8, 5, 7, 9, 10])
    check("postorder", tree.postorder(), [1, 3, 2, 5, 7, 9, 10, 6, 8, 4])
    check("height", tree.height(), 2)
    check("search 7 (present)", tree.search(7), True)
    check("search 11 (absent)", tree.search(11), False)
    tree.insert(5)
    check("inserting duplicate 5 changes nothing", tree.inorder(), list(range(1, 11)))

    # ---------------- Test 2: insertions that cause splits, t = 3 ----------------
    print("\nTest 2: insertions with t = 3 (letters as keys)")
    tree = BTree(t=3)
    tree.root = make_node("GMPX", [make_node("ACDE"), make_node("JK"), make_node("NO"),
                                   make_node("RSTUV"), make_node("YZ")])
    print("  Starting tree:")
    tree.print_tree()

    tree.insert("B")
    print("  Insert B  (goes into a leaf that is not full):")
    tree.print_tree()
    check("insert B", tree.levels(),
          [L("GMPX"), L("ABCDE", "JK", "NO", "RSTUV", "YZ")])

    tree.insert("Q")
    print("  Insert Q  (leaf RSTUV is full -> split, T moves up):")
    tree.print_tree()
    check("insert Q", tree.levels(),
          [L("GMPTX"), L("ABCDE", "JK", "NO", "QRS", "UV", "YZ")])

    tree.insert("L")
    print("  Insert L  (root GMPTX is full -> split, tree grows taller):")
    tree.print_tree()
    check("insert L", tree.levels(),
          [L("P"), L("GM", "TX"), L("ABCDE", "JKL", "NO", "QRS", "UV", "YZ")])

    tree.insert("F")
    print("  Insert F  (leaf ABCDE is full -> split, C moves up):")
    tree.print_tree()
    check("insert F", tree.levels(),
          [L("P"), L("CGM", "TX"), L("AB", "DEF", "JKL", "NO", "QRS", "UV", "YZ")])

    # ---------------- Test 3: deletions covering every delete case, t = 3 ----------------
    print("\nTest 3: deletions with t = 3 (continuing from Test 2)")

    tree.delete("F")
    print("  Delete F  - case 1 (key in a leaf):")
    tree.print_tree()
    check("delete F (case 1)", tree.levels(),
          [L("P"), L("CGM", "TX"), L("AB", "DE", "JKL", "NO", "QRS", "UV", "YZ")])

    tree.delete("M")
    print("  Delete M  - case 2a (replace M by predecessor L):")
    tree.print_tree()
    check("delete M (case 2a)", tree.levels(),
          [L("P"), L("CGL", "TX"), L("AB", "DE", "JK", "NO", "QRS", "UV", "YZ")])

    tree.delete("G")
    print("  Delete G  - case 2c (merge DE + G + JK, then delete G):")
    tree.print_tree()
    check("delete G (case 2c)", tree.levels(),
          [L("P"), L("CL", "TX"), L("AB", "DEJK", "NO", "QRS", "UV", "YZ")])

    tree.delete("D")
    print("  Delete D  - case 3b (merge CL + P + TX, root shrinks, height drops):")
    tree.print_tree()
    check("delete D (case 3b)", tree.levels(),
          [L("CLPTX"), L("AB", "EJK", "NO", "QRS", "UV", "YZ")])

    tree.delete("B")
    print("  Delete B  - case 3a (borrow: C moves down, E moves up):")
    tree.print_tree()
    check("delete B (case 3a)", tree.levels(),
          [L("ELPTX"), L("AC", "JK", "NO", "QRS", "UV", "YZ")])

    check("delete Z2 (absent key) returns False", tree.delete("Z2"), False)
    check("inorder after deletions", tree.inorder(), list("ACEJKLNOPQRSTUVXYZ"))

    # ---------------- Test 4: random test, checked against Python's sorted() ----------------
    print("\nTest 4: 1000 random inserts + 500 random deletes (t = 3), checked against a Python set")
    random_test(BTree(t=3))


# =============================================================================
#                        PART 1b: B+ Tree correctness
# =============================================================================

def test_bplus():
    heading("PART 1b: B+ TREE CORRECTNESS TESTS")

    # ---------------- Test 1: insert 1..10 with order m = 4 ----------------
    print("\nTest 1: insert keys 1..10 into an empty B+ Tree of order m = 4 (max 3 keys per node)")
    tree = BPlusTree(m=4)
    for k in range(1, 11):
        tree.insert(k)
    tree.print_tree()
    check("tree shape", tree.levels(),
          [[[7]], [[3, 5], [9]], [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]])
    check("inorder (leaf chain)", tree.inorder(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    check("preorder",  tree.preorder(),  [7, 3, 5, 1, 2, 3, 4, 5, 6, 9, 7, 8, 9, 10])
    check("postorder", tree.postorder(), [1, 2, 3, 4, 5, 6, 3, 5, 7, 8, 9, 10, 9, 7])
    check("height", tree.height(), 2)
    check("search 7 (present)", tree.search(7), True)
    check("search 11 (absent)", tree.search(11), False)
    check("range_query(4, 8)", tree.range_query(4, 8), [4, 5, 6, 7, 8])
    check("range_query(0, 2)", tree.range_query(0, 2), [1, 2])
    check("range_query(11, 20) (empty)", tree.range_query(11, 20), [])

    # ---------------- Test 2: deletions ----------------
    print("\nTest 2: deletions on the tree from Test 1")

    check("delete 99 (absent key) returns False", tree.delete(99), False)

    tree.delete(7)
    print("  delete 7  - leaf [8] underflows -> merge with [9 10]; parent underflows ->")
    print("              merge with [3 5]; root has one child -> height drops to 1:")
    tree.print_tree()
    check("delete 7 (merge, cascade, root shrinks)", tree.levels(),
          [[[3, 5, 7]], [[1, 2], [3, 4], [5, 6], [8, 9, 10]]])
    print("    (7 is still a separator key in the root - that is allowed in a B+ Tree,")
    print("     separators only guide the search; the data key 7 is gone from the leaves)")
    check("search 7 after delete", tree.search(7), False)

    tree.delete(9)
    print("  delete 9  - leaf still has enough keys, nothing else changes:")
    tree.print_tree()
    check("delete 9 (simple)", tree.levels(),
          [[[3, 5, 7]], [[1, 2], [3, 4], [5, 6], [8, 10]]])

    tree.delete(6)
    print("  delete 6  - leaf [5] underflows -> merge with left sibling [3 4]:")
    tree.print_tree()
    check("delete 6 (merge with left sibling)", tree.levels(),
          [[[3, 7]], [[1, 2], [3, 4, 5], [8, 10]]])

    tree.delete(1)
    print("  delete 1  - leaf [2] underflows -> borrow 3 from right sibling [3 4 5]:")
    tree.print_tree()
    check("delete 1 (borrow from right sibling)", tree.levels(),
          [[[4, 7]], [[2, 3], [4, 5], [8, 10]]])
    check("inorder after deletions", tree.inorder(), [2, 3, 4, 5, 8, 10])
    check("range_query(3, 9) after deletions", tree.range_query(3, 9), [3, 4, 5, 8])

    # ---------------- Test 3: random test ----------------
    print("\nTest 3: 1000 random inserts + 500 random deletes (m = 4), checked against a Python set")
    random_test(BPlusTree(m=4))
    tree = BPlusTree(m=4)
    keys = random.sample(range(5000), 1000)
    for k in keys:
        tree.insert(k)
    check("range_query(1000, 1500) matches a plain filter", tree.range_query(1000, 1500),
          sorted(k for k in keys if 1000 <= k <= 1500))


def random_test(tree):
    """Random inserts/deletes; compare the tree with a Python set (the 'expected' answer)."""
    reference = set()
    keys = random.sample(range(10000), 1000)
    for k in keys:
        tree.insert(k)
        reference.add(k)
    check("inorder == sorted(inserted keys)", tree.inorder(), sorted(reference))
    check("every inserted key is found", all(tree.search(k) for k in keys), True)

    to_delete = random.sample(keys, 500)
    for k in to_delete:
        tree.delete(k)
        reference.remove(k)
    check("inorder == sorted(remaining keys)", tree.inorder(), sorted(reference))
    check("deleted keys are not found", any(tree.search(k) for k in to_delete), False)
    check("remaining keys are found", all(tree.search(k) for k in reference), True)


# =============================================================================
#                        PART 2: Runtime measurements
# =============================================================================

SIZES = [1000, 5000, 10000, 50000, 100000]
NUM_OPS = 1000              # number of timed searches / deletes / range queries (<= smallest size)
REPEATS = 5                 # each measurement is repeated, the fastest run is kept
DEGREE_TEST_N = 20000       # number of keys for the node-size experiment
B_TREE_T = 4                # B-Tree : max 2t   = 8 children per node
B_PLUS_M = 8                # B+ Tree: max m    = 8 children per node (same fan-out)


def measure(tree, n):
    """
    Build a tree with n random keys and time each operation.
    Returns average time per operation in microseconds, plus the tree height.
    """
    keys = list(range(n))
    random.shuffle(keys)

    # Python's garbage collector can pause the program at random moments while
    # thousands of nodes are alive, which adds noise to the timings. Pause it here.
    gc.collect()
    gc.disable()

    start = time.perf_counter()
    for k in keys:
        tree.insert(k)
    insert_us = (time.perf_counter() - start) / n * 1e6

    search_keys = random.sample(keys, NUM_OPS)
    start = time.perf_counter()
    for k in search_keys:
        tree.search(k)
    search_us = (time.perf_counter() - start) / NUM_OPS * 1e6

    range_us = None
    if isinstance(tree, BPlusTree):
        # each query asks for 100 consecutive keys
        starts = [random.randint(0, n - 100) for _ in range(NUM_OPS)]
        start = time.perf_counter()
        for lo in starts:
            tree.range_query(lo, lo + 99)
        range_us = (time.perf_counter() - start) / NUM_OPS * 1e6

    height = tree.height()

    delete_keys = random.sample(keys, NUM_OPS)
    start = time.perf_counter()
    for k in delete_keys:
        tree.delete(k)
    delete_us = (time.perf_counter() - start) / NUM_OPS * 1e6

    gc.enable()
    return {"insert": insert_us, "search": search_us, "delete": delete_us,
            "range": range_us, "height": height}


def fastest(runs):
    """
    Combine several measure() results by keeping the fastest time for every
    operation. Other programs running on the computer can only make a run
    slower, never faster, so the minimum is the most reliable value.
    """
    best = {}
    for op in runs[0]:
        if runs[0][op] is None:              # e.g. range query for a B-Tree
            best[op] = None
        else:
            best[op] = min(run[op] for run in runs)
    return best


def run_benchmarks():
    heading("PART 2: RUNTIME MEASUREMENTS (average microseconds per operation)")
    print(f"B-Tree t = {B_TREE_T},  B+ Tree m = {B_PLUS_M},  "
          f"{NUM_OPS} timed searches/deletes/range queries per size, best of {REPEATS} runs")
    degrees = [2, 4, 8, 16, 32, 64, 128]

    # Every experiment we want to run: (name, value, number of keys)
    jobs = []
    for n in SIZES:
        jobs.append(("B", n, n))
        jobs.append(("B+", n, n))
    for d in degrees:
        jobs.append(("B deg", d, DEGREE_TEST_N))
        jobs.append(("B+ deg", d, DEGREE_TEST_N))

    # We do REPEATS rounds over ALL jobs, in a random order each round (instead of
    # repeating one job several times in a row). If the computer slows down for a
    # while (other programs, CPU getting hot), it hits different jobs in different
    # rounds, and fastest() keeps the good runs.
    runs = {}                                   # (name, value) -> list of measure() results
    for r in range(REPEATS):
        print(f"  round {r + 1} of {REPEATS} ...")
        random.shuffle(jobs)
        for name, value, n in jobs:
            if name == "B":
                tree = BTree(t=B_TREE_T)
            elif name == "B+":
                tree = BPlusTree(m=B_PLUS_M)
            elif name == "B deg":
                tree = BTree(t=value)
            else:
                tree = BPlusTree(m=2 * value)
            runs.setdefault((name, value), []).append(measure(tree, n))

    # ---- table 1: different input sizes ----
    btree_results = []
    bplus_results = []
    print(f"\n{'n':>8} | {'tree':<7} | {'insert':>8} | {'search':>8} | {'delete':>8} | "
          f"{'range100':>8} | {'height':>6}")
    print("-" * 72)
    for n in SIZES:
        b = fastest(runs[("B", n)])
        p = fastest(runs[("B+", n)])
        btree_results.append(b)
        bplus_results.append(p)
        print(f"{n:>8} | {'B-Tree':<7} | {b['insert']:8.2f} | {b['search']:8.2f} | "
              f"{b['delete']:8.2f} | {'-':>8} | {b['height']:>6}")
        print(f"{'':>8} | {'B+Tree':<7} | {p['insert']:8.2f} | {p['search']:8.2f} | "
              f"{p['delete']:8.2f} | {p['range']:8.2f} | {p['height']:>6}")

    # ---- table 2: effect of the node size on a tree with a fixed number of keys ----
    print(f"\nEffect of node size, n = {DEGREE_TEST_N} (B-Tree t = d, B+ Tree m = 2d -> same fan-out)")
    print(f"{'d':>5} | {'B insert':>9} | {'B search':>9} | {'B+ insert':>9} | {'B+ search':>9} | "
          f"{'B h':>4} | {'B+ h':>4}")
    print("-" * 64)
    degree_results = []
    for d in degrees:
        b = fastest(runs[("B deg", d)])
        p = fastest(runs[("B+ deg", d)])
        degree_results.append((b, p))
        print(f"{d:>5} | {b['insert']:9.2f} | {b['search']:9.2f} | {p['insert']:9.2f} | "
              f"{p['search']:9.2f} | {b['height']:>4} | {p['height']:>4}")

    return btree_results, bplus_results, degrees, degree_results


# =============================================================================
#                              PART 3: Plots
# =============================================================================

def make_plots(btree_results, bplus_results, degrees, degree_results):
    heading("PART 3: PLOTS")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"B-Tree (t={B_TREE_T}) vs B+ Tree (m={B_PLUS_M}) - average time per operation")

    # Insert, search, delete: both trees on the same axes
    for ax, op in zip([axes[0][0], axes[0][1], axes[0][2]], ["insert", "search", "delete"]):
        ax.plot(SIZES, [r[op] for r in btree_results], "o-", label="B-Tree")
        ax.plot(SIZES, [r[op] for r in bplus_results], "s-", label="B+ Tree")
        ax.set_xscale("log")
        ax.set_title(op.capitalize())
        ax.set_xlabel("number of keys n (log scale)")
        ax.set_ylabel("microseconds / operation")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Range query (B+ Tree only)
    ax = axes[1][0]
    ax.plot(SIZES, [r["range"] for r in bplus_results], "s-", color="tab:orange", label="B+ Tree")
    ax.set_xscale("log")
    ax.set_title("Range query (100 keys per query)")
    ax.set_xlabel("number of keys n (log scale)")
    ax.set_ylabel("microseconds / query")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Height
    ax = axes[1][1]
    ax.plot(SIZES, [r["height"] for r in btree_results], "o-", label="B-Tree")
    ax.plot(SIZES, [r["height"] for r in bplus_results], "s-", label="B+ Tree")
    ax.set_xscale("log")
    ax.set_title("Tree height")
    ax.set_xlabel("number of keys n (log scale)")
    ax.set_ylabel("height (edges)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Node size experiment
    ax = axes[1][2]
    ax.plot(degrees, [b["search"] for b, p in degree_results], "o-", label="B-Tree search")
    ax.plot(degrees, [p["search"] for b, p in degree_results], "s-", label="B+ Tree search")
    ax.plot(degrees, [b["insert"] for b, p in degree_results], "o--", label="B-Tree insert")
    ax.plot(degrees, [p["insert"] for b, p in degree_results], "s--", label="B+ Tree insert")
    ax.set_xscale("log", base=2)
    ax.set_title(f"Effect of node size (n = {DEGREE_TEST_N})")
    ax.set_xlabel("d  (B-Tree t = d, B+ Tree m = 2d)")
    ax.set_ylabel("microseconds / operation")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("runtime_plots.png", dpi=120)
    print("Plots saved to runtime_plots.png")


# =============================================================================

def main():
    random.seed(42)          # fixed seed so every run gives the same random tests
    test_btree()
    test_bplus()
    heading(f"TEST SUMMARY: {results['passed']} passed, {results['failed']} failed")

    btree_results, bplus_results, degrees, degree_results = run_benchmarks()
    make_plots(btree_results, bplus_results, degrees, degree_results)


if __name__ == "__main__":
    main()
