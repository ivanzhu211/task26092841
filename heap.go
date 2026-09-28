package prosa

// taskHeap is a min-heap of pending tasks ordered by dispatch order:
// higher priority first, then lower task id (earlier submission) first.
type taskHeap []*task

func (h taskHeap) Len() int { return len(h) }

func (h taskHeap) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.priority != b.priority {
		return a.priority > b.priority
	}
	return a.id < b.id
}

func (h taskHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].heapIdx = i
	h[j].heapIdx = j
}

func (h *taskHeap) Push(x any) {
	t := x.(*task)
	t.heapIdx = len(*h)
	*h = append(*h, t)
}

func (h *taskHeap) Pop() any {
	old := *h
	n := len(old)
	t := old[n-1]
	old[n-1] = nil
	t.heapIdx = -1
	*h = old[:n-1]
	return t
}
