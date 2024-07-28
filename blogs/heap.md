---
layout: page
permalink: /blogs/heap/index.html
title: "Understanding Heap Building: O(n log n) vs. O(n)"
---

# Understanding Heap Building

**Date:** July 27, 2024 | **Estimated Reading Time:** 25 min | **Author:** Hector Motsepe

## Introduction 

Priority Queues are commonly implemented using **Heaps**. A heap is a complete binary tree data structure that can be implemented either as a **Min-heap** (where the parent node is always smaller than or equal to its children) or a **Max-heap** (where the parent node is always larger than or equal to its children). This structure allows us to represent the heap as an array, where for a node at index `i`:

- Its left child is at index `2i + 1`
- Its right child is at index `2i + 2`
- Its parent is at index `(i - 1) / 2`

For instance, we can convert the following tree into an array structure (and vice versa): `[15, 10, 9, 7, 8, 3, 5]`:

             15
            /  \
           10   9
          /  \ /  \
         7   8 3   5

- The root at index 0 -> `[15]`
- Its left child (10) is at index (2*0 + 1) -> `[15, 10]`
- Its right child (9) is at index (2*0 + 2) -> `[15, 10, 9]`
- Left child of (10): at index (2*1 + 1) -> `[15, 10, 9, 7]`
- Right child of (10): at index (2*1 + 2) -> `[15, 10, 9, 7, 8]`
- Left child of (9): at index (2*2 + 1) -> `[15, 10, 9, 7, 8, 3]`
- Right child of (9): at index (2*2 + 2) -> `[15, 10, 9, 7, 8, 3, 5]`

In a previous [blog post](https://motsepe-jr.github.io/blogs/dijkstra/), we explored Dijkstra's Algorithm, which employs a priority queue to process vertices based on their current known shortest distance. This approach enhances the time complexity to `O((V + E) log V)`, compared to using a normal queue data structure, which results in `O(V^2)` due to the need to search for the minimum distance vertex in each iteration.

Inserting elements one at a time into a heap data structure results in `O(n log n)` time complexity. In this blog, I would like to discuss why building (or initializing) a heap with 'N' elements will take `O(N)`, whereas if we insert elements one by one into the heap, it will take `O(N log N)`.

## The O(N log N) Heap Building Approach

First, let's look at the scenario where the heap building algorithm takes `O(n log n)`. Whenever we add elements one at a time, we need to "bubble up" each element to its correct position. In the worst-case scenario, the new element might need to move all the way up to the root, which takes `O(log n)` steps in a binary tree with n nodes. Therefore, because we're adding one element at a time, after we have added all the elements, the overall time complexity is `O(n log n)` -> n is all the elements we added, and log n is the time it takes to bubble each element to the correct position. 

As illustrated in the Rust code below, when we call `heap.push(number)`, the `push` method adds the number to a vector and then calls the `heapify_up` method to bubble up the number to the correct position. In the worst-case scenario, the number will be bubbled up to the root or until the condition `index > 0` becomes false.

```rust
use std::cmp::Ordering;

pub struct MinHeap<T: Ord> {
    heap: Vec<T>,
}

impl<T: Ord> MinHeap<T> {
    pub fn new() -> Self {
        MinHeap { heap: Vec::new() }
    }

    pub fn push(&mut self, item: T) {
        self.heap.push(item);
        self.heapify_up(self.heap.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            None
        } else {
            let last = self.heap.pop().unwrap();
            if !self.heap.is_empty() {
                let first = std::mem::replace(&mut self.heap[0], last);
                self.heapify_down(0);
                Some(first)
            } else {
                Some(last)
            }
        }
    }

    pub fn peek(&self) -> Option<&T> {
        self.heap.first()
    }

    fn heapify_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.heap[index].cmp(&self.heap[parent]) == Ordering::Less {
                self.heap.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn heapify_down(&mut self, mut index: usize) {
        let len = self.heap.len();
        loop {
            let left_child = 2 * index + 1;
            let right_child = 2 * index + 2;
            let mut smallest = index;

            if left_child < len && self.heap[left_child].cmp(&self.heap[smallest]) == Ordering::Less {
                smallest = left_child;
            }
            if right_child < len && self.heap[right_child].cmp(&self.heap[smallest]) == Ordering::Less {
                smallest = right_child;
            }

            if smallest != index {
                self.heap.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }
}

fn main() {
    let vec = vec![10, 5, 3, 4, 1];
    println!("Original vector: {:?}", vec);

    let mut heap = MinHeap::new();

    heap.push(2);
    println!("After pushing 2: {:?}", heap.peek());

    heap.push(5);
    println!("After pushing 5: {:?}", heap.peek());

    heap.push(1);
    println!("After pushing 1: {:?}", heap.peek());

    heap.push(3);
    println!("After pushing 3: {:?}", heap.peek());

    heap.push(4);
    println!("After pushing 4: {:?}", heap.peek());

    while let Some(min) = heap.pop() {
        print!("{} ", min);
    }
    println!("\nHeap is empty: {:?}", heap.peek());
}
```

## The O(N) Heap Building Approach

The heap building algorithm we discussed earlier is efficient for streaming data. However, what if we already have the data in an array? Let's explore the O(N) heap building algorithm.

When we build (initialize) the heap from an unsorted array of elements, the time complexity becomes O(N). We treat the array as if it were already a complete binary tree in terms of structure, just not in terms of the heap property. Starting from the parent of the last element, we work our way up to the root. For each element, we perform the **heapify down** operation to ensure that the subtree rooted at the node satisfies the heap property.

### Why O(N)?

We only heapify the non-leaf nodes, which at most are n/2 nodes. The heapify operation takes more time for nodes closer to the root and less time for nodes closer to the leaves. Consider the following tree structure:

                   1
                /     \
               2       3
             /   \    /   \
            4     5  6      7
          /   \  /   \    /   \
        8     9 10    11 12     14

At each height:
-  there are n/2^1 (here 13/2 ≈ 7) nodes
-  there are n/2^2 (here 13/4 ≈ 3) nodes
-  there are n/2^3 (here 13/8 ≈ 2) nodes
-  there are n/2^4 (here 13/16 ≈ 1) node

So, there are n/2^(h+1) nodes for height h.

To find the time complexity, let's count the amount of work done or max number of iterations performed by each node:

- Nodes at height 0: n/2^1 * 0 (zero since no children)
- Nodes at height 1: n/2^2 * 1 (heapify will perform at most one swap for each node)
- Nodes at height 2: n/2^3 * 2 (heapify will perform at most two swaps for each node)
- Nodes at height 3: n/2^4 * 3 (heapify will perform at most three swaps for each node)

For any nodes with height h, maximum work done is n/2^(h+1) * h. We sum the total work for nodes at each height:

(n/2^1 * 0) + (n/2^2 * 1) + (n/2^3 * 2) + (n/2^4 * 3) + ... + (n/2^(h+1) * h)

We can simplify this into n * (1/4 + 2/8 + 3/16 + ...). The sum in parentheses converges to a constant (about 1). Thus, the time complexity of building the heap is O(n).

Here's the implementation of the MinHeap with the O(N) building approach:

```rust
pub struct MinHeap<T: Ord> {
    heap: Vec<T>,
}

impl<T: Ord> MinHeap<T> {
    pub fn new() -> Self {
        MinHeap { heap: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        MinHeap { heap: Vec::with_capacity(capacity) }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut heap = MinHeap { heap: vec };
        heap.build_heap();
        heap
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn push(&mut self, item: T) {
        self.heap.push(item);
        self.heapify_up(self.heap.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            None
        } else {
            let last = self.heap.pop().unwrap();
            if !self.heap.is_empty() {
                let first = std::mem::replace(&mut self.heap[0], last);
                self.heapify_down(0);
                Some(first)
            } else {
                Some(last)
            }
        }
    }

    pub fn peek(&self) -> Option<&T> {
        self.heap.first()
    }

    fn build_heap(&mut self) {
        if self.heap.is_empty() {
            return;
        }
        let last_parent = (self.heap.len() - 2) / 2;
        for i in (0..=last_parent).rev() {
            self.heapify_down(i);
        }
    }

    fn heapify_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.heap[index] < self.heap[parent] {
                self.heap.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    fn heapify_down(&mut self, mut index: usize) {
        let len = self.heap.len();
        loop {
            let left_child = 2 * index + 1;
            let right_child = 2 * index + 2;
            let mut smallest = index;

            if left_child < len && self.heap[left_child] < self.heap[smallest] {
                smallest = left_child;
            }
            if right_child < len && self.heap[right_child] < self.heap[smallest] {
                smallest = right_child;
            }

            if smallest != index {
                self.heap.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }
}
fn main() {

    //=============0(n log n) Approach============
    let vec = vec![10, 5, 3, 4, 1];
    println!("Original vector: {:?}", vec);

    let mut heap = MinHeap::new();

    heap.push(2);
    println!("After pushing 2: {:?}", heap.peek());

    heap.push(5);
    println!("After pushing 5: {:?}", heap.peek());

    heap.push(1);
    println!("After pushing 1: {:?}", heap.peek());

    heap.push(3);
    println!("After pushing 3: {:?}", heap.peek());

    heap.push(4);
    println!("After pushing 4: {:?}", heap.peek());

    while let Some(min) = heap.pop() {
        print!("{} ", min);
    }
    println!("\nHeap is empty: {:?}", heap.peek());

    // =================0(n) Approach================
    let vec = vec![10, 5, 3, 4, 1];
    println!("Original vector: {:?}", vec);

    let mut heap = MinHeap::from_vec(vec);
    println!("After building heap: {:?}", heap.heap);

    heap.push(2);
    println!("After pushing 2: {:?}", heap.heap);

    while let Some(min) = heap.pop() {
        print!("{} ", min);
    }
    println!("\nHeap is empty: {}", heap.is_empty());
}
```

## Conclusion

In this blog post, we explored the O(N) heap building algorithm, which is more efficient than the O(N log N) approach when we have all the data available upfront. We discussed the mathematical reasoning behind its time complexity and provided a Rust implementation of a MinHeap with this efficient building method.
The key takeaways are:

The O(N) approach treats the input array as a complete binary tree.
It performs heapify-down operations starting from the last non-leaf node up to the root.
The time complexity is O(N) due to the decreasing work required for nodes closer to the leaves.

This efficient heap building algorithm is crucial in various applications, such as heap sort and priority queues, where performance is critical.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
- Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley Professional.
- Rust Documentation. (n.d.). std::collections::BinaryHeap. https://doc.rust-lang.org/std/collections/struct.BinaryHeap.html
- Stackoerflow: https://stackoverflow.com/questions/9755721/how-can-building-a-heap-be-on-time-complexity