import numpy as np

class VirtualKVCacheManager:
    def __init__(self, num_physical_blocks: int, block_size: int, hidden_dim: int):
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.k_blocks = np.zeros(
            (num_physical_blocks, block_size, hidden_dim), dtype=float
        )
        self.v_blocks = np.zeros(
            (num_physical_blocks, block_size, hidden_dim), dtype=float
        )
        self.free_blocks = list(range(num_physical_blocks))

        self.page_tables = {}


    def allocate_block(self, seq_id: int) -> int:
        """Allocate a new physical block for seq_id. Return logical index or -1."""
        if not self.free_blocks:
            return -1

        physical_block = self.free_blocks.pop(0)
        page_table = self.page_tables.setdefault(seq_id, [])
        page_table.append(physical_block)

        return len(page_table) - 1

    def write(self, seq_id, position, key, value):
        if position < 0:
            return False

        key = np.asarray(key, dtype=float)
        value = np.asarray(value, dtype=float)

        if key.shape != (self.hidden_dim,) or value.shape != (self.hidden_dim,):
            return False

        logical_block = position // self.block_size
        slot = position % self.block_size

        page_table = self.page_tables.setdefault(seq_id, [])

        # 补充分配逻辑块，直到覆盖目标位置
        while len(page_table) <= logical_block:
            if self.allocate_block(seq_id) == -1:
                return False
            page_table = self.page_tables[seq_id]

        physical_block = page_table[logical_block]
        self.k_blocks[physical_block, slot] = key
        self.v_blocks[physical_block, slot] = value
        return True

    def read(self, seq_id, position):
        if position < 0 or seq_id not in self.page_tables:
            return None, None

        logical_block = position // self.block_size
        slot = position % self.block_size
        page_table = self.page_tables[seq_id]

        if logical_block >= len(page_table):
            return None, None

        physical_block = page_table[logical_block]
        return (
            self.k_blocks[physical_block, slot].copy(),
            self.v_blocks[physical_block, slot].copy(),
        )

    def free_sequence(self, seq_id):
        page_table = self.page_tables.pop(seq_id, [])
        for physical_block in page_table:
            self.k_blocks[physical_block].fill(0)
            self.v_blocks[physical_block].fill(0)
            self.free_blocks.append(physical_block)

        self.free_blocks.sort()
        return len(page_table)

    def get_num_free_blocks(self):
        return len(self.free_blocks)

    def get_page_table(self, seq_id):
        return self.page_tables.get(seq_id, []).copy()