def analyze_memory_fragmentation(block_status: list) -> dict:
	"""
	Analyze memory fragmentation in a block-based memory pool.

	Args:
		block_status: List of ints where 1 = allocated block, 0 = free block

	Returns:
		dict with keys:
			'utilization': float, fraction of allocated blocks
			'num_free_fragments': int, count of contiguous free regions
			'largest_free_fragment': int, size of largest contiguous free region
			'fragmentation_ratio': float, measure of free memory scatter
	"""
	prev = 1
	num_fra = 0
	max_fra = 0
	num_alloc = 0
	curr_fra = 0
	for i in range(len(block_status)):
		if prev==0:
			if block_status[i] == 0:
				curr_fra+=1
				max_fra = max(max_fra,curr_fra)
			else:
				num_alloc += 1
				prev = 1
				curr_fra = 0
		else:
			if block_status[i]==0:
				curr_fra+=1
				prev=0
				max_fra = max(max_fra,curr_fra)
				num_fra+=1
			else:
				num_alloc+=1
				prev=1
				curr_fra=0
	utilization = num_alloc/len(block_status)
	num_free_fragments = num_fra
	largest_free_fragment = max_fra
	fragmentation_ratio = 0
	if (len(block_status)-num_alloc) == 0:
		fragmentation_ratio = 0
	else:
		fragmentation_ratio = 1 - (largest_free_fragment/(len(block_status)-num_alloc))
	ans = {'utilization': utilization, 'num_free_fragments': num_free_fragments, 'largest_free_fragment': largest_free_fragment, 'fragmentation_ratio': fragmentation_ratio}
	return ans


