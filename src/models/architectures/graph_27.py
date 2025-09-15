import numpy as np

class Graph:
    def __init__(self, strategy: str = 'uniform', num_node: int = 27):
        self.num_node = num_node
        # Selected joints from notebook mapping (original indices -> 27 nodes)
        selected = np.concatenate((
            [0, 5, 6, 7, 8, 9, 10],
            [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
            [112, 116, 117, 120, 121, 124, 125, 128, 129, 132]
        ))
        self.selected = selected.tolist()
        self.index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.selected)}

        pairs = [
            (5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 91),
            (91, 95), (91, 96), (91, 99), (91, 100),
            (95, 96), (99, 100), (103, 104), (107, 108),
            (112, 116), (112, 117), (112, 120), (112, 121),
            (116, 117), (120, 121), (124, 125), (128, 129),
            (10, 112), (9, 91)
        ]

        mapped_pairs = []
        for i, j in pairs:
            if i in self.index_mapping and j in self.index_mapping:
                mapped_pairs.append((self.index_mapping[i], self.index_mapping[j]))

        self.inward = mapped_pairs
        self.outward = [(j, i) for (i, j) in self.inward]
        self.self_links = [(i, i) for i in range(self.num_node)]
        self.A = self.get_adjacency()

    def edge2mat(self, link, num_node):
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in link:
            if 0 <= i < num_node and 0 <= j < num_node:
                A[j, i] = 1
        return A

    def normalize_digraph(self, A):
        Dl = A.sum(0)
        D = np.zeros_like(A)
        for i in range(A.shape[1]):
            if Dl[i] > 0:
                D[i, i] = Dl[i] ** -1
        return A @ D

    def get_adjacency(self):
        A = np.stack([
            self.edge2mat(self.self_links, self.num_node),
            self.normalize_digraph(self.edge2mat(self.inward, self.num_node)),
            self.normalize_digraph(self.edge2mat(self.outward, self.num_node))
        ])
        return A.astype(np.float32)


