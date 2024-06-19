import math


class HeapCluster():

    def __init__(self, tuple_list):
        """
        tuple_list: list of tuples
            for each tuple in tuple_list
                argument 0 of tuple_list should be a distance
                argument 1 of tuple_list should be a cluster label A
                argument 2 of tuple_list should be a cluster label B
        """
        self.heap = tuple_list
        self.heap_index = {}
        self.heap_size = 0

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

    def parent(self, i):

        return (i - 1) // 2

    def swap(self, index_a, index_b):
        temp = self.heap[index_a]
        self.heap[index_a] = self.heap[index_b]
        self.heap[index_b] = temp

    def swap_dict(self, index_a, index_b):
        key_a = str(self.heap[index_a][1]) + str(self.heap[index_a][2])
        key_b = str(self.heap[index_b][1]) + str(self.heap[index_b][2])
        temp = self.heap_index[key_a]  # temp == index_a
        self.heap_index[key_a] = self.heap_index[key_b]
        self.heap_index[key_b] = temp

    def min_heapify_down(self, index):
        left = self.left(index)
        right = self.right(index)
        if left < self.heap_size and self.heap[left][0] < self.heap[index][0]:
            min = left
        else:
            min = index
        if right < self.heap_size and self.heap[right][0] < self.heap[min][0]:
            min = right
        if min != index:
           # self.swap_dict(index, min)
            self.swap(index, min)
            self.min_heapify_down(min)

    def min_heapify_up(self, index):
        print()
        i = index
        value = self.heap[index][0]
        while i > 0 and value < self.heap[self.parent(i)][0]:
            #self.swap_dict(i, self.parent(i))
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def init_heap_index(self):
        for i, triplet_of_values in zip(range(len(self.heap)), self.heap):
            key = str(triplet_of_values[1]) + str(triplet_of_values[2])
            self.heap_index[key] = i

    def build_heap(self):
        self.heap_size = len(self.heap) - 1
        for i in range(self.parent(self.heap_size), -1, -1):
            self.min_heapify_down(i)

    def insert_k_mins(self, distance_tuple, k):
        if self.heap_size +1 < k:
            self.insert(distance_tuple)
        else:
            if distance_tuple[0] < self.heap[0][0]:
                self.heap[self.heap_size] = distance_tuple
                self.min_heapify_up(self.heap_size)
            else:
                return

    def insert(self, distance_tuple):
        #print("BEFORE INSERT", self.heap)
        self.heap.append(distance_tuple)
        #print("AFTER INSERT", self.heap)
        self.heap_size = len(self.heap) - 1
        #key = str(distance_tuple[1]) + str(distance_tuple[2])
       # self.heap_index[key] = self.heap_size
        self.min_heapify_up(self.heap_size)

    def remove(self, label_a, label_b):
        key = str(label_a) + str(label_b)
        if key not in self.heap_index:
            # print(f"KEY {key} IS NOT HERE")
            return
        # print(f"KEY {key} IS HERE")
        index_to_remove = self.heap_index[key]
        self.swap_dict(index_to_remove, self.heap_size)
        self.swap(index_to_remove, self.heap_size)
        value = self.heap[index_to_remove][0]
        self.heap.pop()
        self.heap_size -= 1

        if self.heap_size == 0:
            del self.heap_index[key]
            return

        left = self.left(index_to_remove)
        right = self.right(index_to_remove)
        parent = self.parent(index_to_remove)

        if left < self.heap_size and value > self.heap[left][0]:
            self.min_heapify_down(index_to_remove)
        elif right < self.heap_size and value > self.heap[right][0]:
            self.min_heapify_down(index_to_remove)
        elif parent >= 0 and value < self.heap[parent][0]:
            self.min_heapify_up(index_to_remove)

        del self.heap_index[key]

    def pop(self):

        if len(self.heap) == 0:
            return (float('inf' , None , None))

        result = self.heap[0]
        key_to_remove = str(self.heap[0][1]) + str(self.heap[0][2])
       # self.swap_dict(0, self.heap_size)
        self.swap(0, self.heap_size)
        self.heap.pop()
        self.heap_size = len(self.heap) - 1
        self.min_heapify_down(0)
        #del self.heap_index[key_to_remove]
        return result

    def print_heap(self):
        n = len(self.heap)
        levels = math.floor(math.log2(n)) + 1 if n > 0 else 0
        max_width = 2 ** (levels - 1) if levels > 0 else 0
        max_label_width = max(len(str(item)) for item in self.heap) if self.heap else 0

        for level in range(levels):
            level_width = 2 ** level
            level_items = self.heap[2 ** level - 1:2 ** (level + 1) - 1]
            spaces_between = (max_width - level_width) * max_label_width
            spaces_before = (max_width - level_width) * max_label_width // 2
            print(" " * spaces_before, end="")
            for i, item in enumerate(level_items):
                print(f"{str(item):^{max_label_width}}", end="")
                if i < len(level_items) - 1:
                    print(" " * spaces_between, end="")
            print()

'''
if __name__ == '__main__':
    
    tripletas = [
        (5, 'A', 'B'), #0
        (2, 'A', 'C'), #1
        (8, 'B', 'D'), #2
        (1, 'C', 'E'), #3
        (3, 'D', 'E'), #4
        (4, 'A', 'D'), #5
        (7, 'B', 'C'), #6
        (6, 'C', 'D') #7
    ]

heap_c = HeapCluster([])
heap_c.insert_k_mins(tripletas[5], 3)
heap_c.insert_k_mins(tripletas[6], 3)
heap_c.insert_k_mins(tripletas[2], 3)
print(heap_c.heap)
heap_c.insert_k_mins(tripletas[3], 3)
print(heap_c.heap)
'''