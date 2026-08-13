"""
btree_demo.py — B-Tree Demonstration Script
  1. Reads test cases from btree_test_cases.txt and verifies correctness.
  2. Measures runtime of insert, search, and delete on large random inputs.
  3. Plots runtime vs. input size and saves as btree_plot.png.
"""

import time
import random
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from btree import BTree


# ================================================================ #
#  PART 1 — Correctness Verification (small test cases)            #
# ================================================================ #
print("=" * 60)
print("  B-TREE — Correctness Verification")
print("=" * 60)

# Read test data from file
with open("btree_test_cases.txt", "r") as f:
    data = [int(x.strip()) for x in f.read().split(",")]

print(f"\nInput data       : {data}")

# Build a B-Tree with minimum degree t = 3 (max 5 keys per node)
tree = BTree(t=3)
for key in data:
    tree.insert(key)

# Verify via in-order traversal
print(f"In-order traversal: {tree.traverse()}")

# Search tests — mix of present and absent keys
print("\n--- Search Tests ---")
for key in [6, 12, 30, 99]:
    status = "FOUND" if tree.search(key) else "NOT FOUND"
    print(f"  search({key:>3d})  ->  {status}")

# Delete tests — remove two keys and show the tree after each
print("\n--- Delete Tests ---")
for key in [6, 30]:
    tree.delete(key)
    print(f"  Deleted {key:>3d}  ->  Traversal: {tree.traverse()}")

# Confirm the deleted keys are truly gone
print("\n--- Post-Delete Search ---")
for key in [6, 30]:
    status = "FOUND" if tree.search(key) else "NOT FOUND"
    print(f"  search({key:>3d})  ->  {status}")


# ================================================================ #
#  PART 2 — Runtime Measurements                                   #
# ================================================================ #
print("\n" + "=" * 60)
print("  B-TREE — Runtime Measurements")
print("=" * 60)

sizes = [10_000, 50_000, 100_000]
insert_times = []
search_times = []
delete_times = []

for n in sizes:
    # Generate n unique random integers
    keys = random.sample(range(1, n * 10), n)

    # --- Measure Insertion ---
    tree = BTree(t=50)                       # Higher degree for realistic fan-out
    start = time.perf_counter()
    for k in keys:
        tree.insert(k)
    t_insert = time.perf_counter() - start
    insert_times.append(t_insert)

    # --- Measure Search (all inserted keys, shuffled) ---
    random.shuffle(keys)
    start = time.perf_counter()
    for k in keys:
        tree.search(k)
    t_search = time.perf_counter() - start
    search_times.append(t_search)

    # --- Measure Deletion (delete all keys) ---
    random.shuffle(keys)
    start = time.perf_counter()
    for k in keys:
        tree.delete(k)
    t_delete = time.perf_counter() - start
    delete_times.append(t_delete)

    print(f"  n = {n:>7,d}  |  Insert: {t_insert:.4f}s  |  "
          f"Search: {t_search:.4f}s  |  Delete: {t_delete:.4f}s")


# ================================================================ #
#  PART 3 — Visualization (saved to btree_plot.png)                #
# ================================================================ #
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(sizes, insert_times, "o-",  color="#e74c3c", label="Insert",
        linewidth=2, markersize=8)
ax.plot(sizes, search_times, "s--", color="#2980b9", label="Search",
        linewidth=2, markersize=8)
ax.plot(sizes, delete_times, "D-.", color="#27ae60", label="Delete",
        linewidth=2, markersize=8)

ax.set_xlabel("Input Size (n)", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("B-Tree — Runtime vs Input Size", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("btree_plot.png", dpi=150)
print(f"\n  Plot saved -> btree_plot.png")
