"""
bplus_demo.py — Simplified B+ Tree Demonstration Script
Reads from the 10 generated test case files, measures runtime, and plots the graph.
"""

import time
import random
import matplotlib.pyplot as plt
from bplus_tree import BPlusTree

print("=" * 60)
print("  B+ TREE — Simplified Runtime Measurements")
print("=" * 60)

sizes = [10, 50, 100, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
insert_times = []
search_times = []
delete_times = []
range_times  = []

for i, n in enumerate(sizes, 1):
    filename = f"bplus_tc_{i}.txt"
    
    with open(filename, "r") as f:
        keys = [int(x) for x in f.read().split(",")]

    tree = BPlusTree(order=50)
    
    # --- Measure Insertion ---
    start = time.perf_counter()
    for k in keys:
        tree.insert(k)
    t_insert = time.perf_counter() - start
    insert_times.append(t_insert)

    # --- Measure Search ---
    random.shuffle(keys)
    start = time.perf_counter()
    for k in keys:
        tree.search(k)
    t_search = time.perf_counter() - start
    search_times.append(t_search)

    # --- Measure Range Query ---
    start = time.perf_counter()
    for _ in range(1000):
        lo = random.randint(1, max(1, n * 10 - 100))
        tree.range_query(lo, lo + 100)
    t_range = time.perf_counter() - start
    range_times.append(t_range)

    # --- Measure Deletion ---
    random.shuffle(keys)
    start = time.perf_counter()
    for k in keys:
        tree.delete(k)
    t_delete = time.perf_counter() - start
    delete_times.append(t_delete)

    print(f"  File: {filename:<20} | Insert: {t_insert:.4f}s | Search: {t_search:.4f}s | Delete: {t_delete:.4f}s | Range: {t_range:.4f}s")


# --- Visualization ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sizes, insert_times, "o-",  color="#e74c3c", label="Insert", linewidth=2)
ax.plot(sizes, search_times, "s--", color="#2980b9", label="Search", linewidth=2)
ax.plot(sizes, delete_times, "^:",  color="#8e44ad", label="Delete", linewidth=2)
ax.plot(sizes, range_times,  "D-.", color="#27ae60", label="Range Query", linewidth=2)

ax.set_xlabel("Input Size (n)", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("B+ Tree — Runtime vs Input Size (File Inputs)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bplus_plot.png", dpi=150)
print(f"\n  Plot saved -> bplus_plot.png")
plt.show()
