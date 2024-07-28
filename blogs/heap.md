---
layout: page
permalink: /blogs/gelu/index.html
title: "Understanding Heap Building: O(n log n) vs. O(n)"
---

# Understanding Heap Building

**Date:** July 27, 2024 | **Estimated Reading Time:** 25 min | **Author:** Hector Motsepe

## Introduction 
Priority Queue are implemented using Heap. A heap is a complete binary tree data structure that can be implemented either as a Min-heap (the parent node is always smaller than or equals to the children) or Max-heap (the parent node is always larger than or equal to the children). This structure allows us to represent the heap as an array, where for a node at index i:

- Its left child is at index 2i + 1
- Its right child is at index 2i + 2
- Its parent is at index (i - 1) / 2

For instances, we can convert the below tree into an array structure (vice versa): [15, 10, 9, 7, 8, 3, 5]:

      15
     /  \
    10   9
   / \  / \
  7  8 3    5

1. the root at index 0 -> [15]
2. Its left child (10) is at index (2*0 + 1) -> [15, 10].
3. Its right child (9) is at index (2*0 + 2) -> [15, 10, 9]
4. Left child of (10): at index (2*1 + 1) -> [15, 10, 9, 7]
5. Right child of (10): at index (2*1 + 2) -> [15, 10, 9, 7, 8]
6. Left child of (9): at index (2*2 + 1) -> [15, 10, 9, 7, 8, 3]
7. Right child of (9): at index (2*2 + 2) -> [15, 10, 9, 7, 8, 3, 5]


In a previous [blog post](https://motsepe-jr.github.io/blogs/dijkstra/), we explored Dijkstra's Algorithm, which employs a priority queue to process vertices based on their current known shortest distance. This approach enhances the time complexity to O((V + E) log V), compared to using a normal queue data structure, which results in O(V^2) due to the need to search for the minimum distance vertex in each iteration. Inserting elements one at the time inside a heap data structure, results in O(n log n) time complexity. In this blog, I would like to discuss why building (or initializing) a heap with 'N' elements will take O(N) whereas if we insert elements one by one in the heap it will take O(N*logN).

## The 0(N log N) Heap Building Approach

First lets look at the scenario where the heap building algorithm takes O(n* log n). Whenever we add elements one at the time, we need to "bubble it up" to its right position. IN the worst case scenario the new element might need to move all the way up to the root which takes O(log n) steps in a binary tree with n nodes. Therefore, because we adding one element at the time, after we have added all the elements the overall time complexity is O(n log n) -> n is all the elements we added, and log n is the time it take to bubble the element to the correct position. 

As illustrated in the below rust code, as we call "heap.push(number)", the push method add the number to a vector and then calls heapify_up method to bubble-up the number to the correct position. In the  worst case scenario the number will be bubbkled up to the root or until the (while index > 0) becomes false. 


===== Rust code ====
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

The above heap building algorithm is good for streaming data, however what if we already the data in an array? the following expand the 0(N) heap building algorithm
## The 0(N) Heap Building Approach

Now when we build (Initialise) the heap from an unsorted array of elements, the time complexity becomes 0(N). We need to treat the array as if it were already a complete binary tree in terms of structure, just not in terms of the heap property. Starting from the parent of the last element, and work your way up to the root. For each element perform the heapify down operation to ensure that then subtree rooited ay the node satifies the heap property.

Why 0(N)? we only heapify the non-leaf nodes, which at most are n/2 nodes. The heapify takes more time for nodes closer to the root, and less time for nodes closer to the leaves.  Consider the below below tree data structure:

         1
      /     \
    2         3
   / \       / \
  4   5     6   7
 / \ / \   / \
8 9 10  11 12 14
8
Height 3:  1                (1 node)
Height 2:  2 3              (2 nodes)
Height 1:  4 5 6 7          (4 nodes)
Height 0:  8 9 10 11 12 14  (6 nodes)

At height:
- 0 there are  n/2^1 (here 13/2 = 6.5 ~7) nodes
- 1 there are n/2^2 (here 13/3 = 3.25 ~3) nodes
- 2 there are n/2^3 (here 13/8 = 1.6 ~ 2) nodes
- 3 ther are n/2^4 (here 13/16 = 0.8 ~ 1) nodes
so there are n/2^(h+1) nodes for height h

To find the time complexity lets count the amount of work done or max no of iterations performed by each node.

Nodes at height 0   = n/2^1 * 0 (zero since no children)  
Nodes at height 1   = n/2^2 * 1 (heapify will perform atmost one swap for each node)  
Nodes at height 2   = n/2^3 * 2 (heapify will perform atmost two swaps for each node)  
Nodes at height 3   = n/2^4 * 3 (heapify will perform atmost three swaps for each node)  

so for any nodes with height h maximum work done is n/2^(h+1) * h. We sum the total work for nodes at each height. 

(n/2^1 * 0) + (n/2^2 * 1)+ (n/2^3 * 2) + (n/2^4 * 3) +...+ (n/2^(h+1) * h) can simplify this into  ( 0 + 1/4 + 2/8 + 3/16 +...+ h/2^(h+1) ) sequence.

the sequence will never execeed on1, thus the time complexity of bulduing the heap is 0(n)

Conclsution


References: