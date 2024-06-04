from __future__ import annotations
from typing import List, Set, Tuple
from collections import deque
import itertools


class Partition:
    def __init__(self, *args):
        args = [int(v) for v in args]
        self._check_args(args)
        self._value = tuple(sorted(args, reverse=True))
        
    def _check_args(self, args):
        for v in args:
            assert v >= 1
        
    def __repr__(self):
        return "Partition{}".format(self._value)
        
    def __hash__(self):
        return hash(self._value)
    
    def __eq__(self, other: Partition):
        return self._value == other._value
    
    def __iter__(self):
        return iter(self._value)

    def aslist(self):
        return list(self._value)

    def astuple(self):
        return self._value

    @property
    def size(self):
        return sum(self._value)

    @property
    def length(self):
        return len(self._value)

    def __getitem__(self, index):
        return self._value[index]
    
    def conjugate(self):  # TODO: optimize
        new_value = []
        for i in range(1, sum(self._value)+1):
            new_value_i = 0
            for v in self._value:
                if v >= i:
                    new_value_i += 1
            if new_value_i == 0:
                break
            else:
                new_value.append(new_value_i)
        return Partition(*new_value)


class Permutation:
    def __init__(self, *args):
        args = [int(v) for v in args]
        self._check_args(args)
        self._value = tuple(args)
        
    def _check_args(self, args):
        n = len(args)
        assert sorted(args) == list(range(1, n+1))
        
    def __repr__(self):
        return "Permutation{}".format(self._value)
    
    def __hash__(self):
        return hash(self._value)
    
    def __eq__(self, other: Permutation):
        return self._value == other._value
    
    def __len__(self):
        return len(self._value)
    
    def __mul__(self, other: Permutation):
        if type(other) != Permutation:
            return NotImplemented
        n = len(self._value)
        assert n == len(other._value)
        new_value = []
        for i in range(n):
            new_value.append(self._value[other._value[i] - 1])
        return Permutation(*new_value)
    
    def sign(self):
        inv = 0
        for i in range(len(self._value)):
            for j in range(i):
                if self._value[j] > self._value[i]:
                    inv += 1
        return (-1)**inv
    
    def inverse(self):
        inv = [0 for _ in range(len(self._value))]
        for i, x in enumerate(self._value):
            inv[x-1] = i+1
        return Permutation(*inv)

    def partition(self) -> Partition:
        visited = set()
        cycle_lens = []
        for k in range(1, self.__len__()+1):
            if k in visited:
                continue
            cycle_lens.append(0)
            while k not in visited:
                cycle_lens[-1] += 1
                visited.add(k)
                k = self._value[k-1]
        return Partition(*cycle_lens)


class Tableau:
    def __init__(self, diagram: Partition, values: Permutation):
        self._diagram = diagram
        self._values = values
        self.check_args()
        
    def check_args(self):
        assert len(self._values) == self._diagram.size
        
    def __hash__(self):
        return hash((self._diagram, self._values))
    
    def __rmul__(self, perm: Permutation):
        return Tableau(
            diagram = self._diagram,
            values = perm * self._values # Other order?)
        )

    def get_diagram(self):
        return self._diagram
    
    def get_columns(self) -> Tuple[Set]: 
        n_cols = self._diagram._value[0]
        cols = [set() for _ in range(n_cols)]
        
        current_col = 0
        current_row = 0
        for i in self._values._value:
            cols[current_col].add(i)
            current_col += 1
            if current_col == self._diagram._value[current_row]:
                current_row += 1
                current_col = 0
                
        return tuple(cols)


class Matching:
    def __init__(self, *value):
        self._value = [tuple(sorted(pair)) for pair in value] # tuple of pairs
        self._value = tuple(sorted(self._value))
        for pair in self._value:
            assert type(pair) == tuple
            assert len(pair) == 2

    def get_number_of_pairs(self):
        return len(self._value)

    def iterate_pairs(self):
        for pair in self._value:
            yield pair
        
    def __eq__(self, other: Matching):
        return self._value == other._value

    def __hash__(self):
        return hash(self._value)
    
    def __repr__(self):
        return "Matching{}".format(self._value)
        
    def __rmul__(self, perm: Permutation):
        assert (2*len(self._value)) == len(perm)
        new_matching = tuple((perm._value[p[0]-1], perm._value[p[1]-1]) for p in self._value)
        return Matching(*new_matching)


def matching_distance(x: Matching, y: Matching) -> Partition:
    assert x.get_number_of_pairs() == y.get_number_of_pairs()
    n = x.get_number_of_pairs()

    # Construct graph
    graph = [[] for _ in range(2 * n)]
    visited = [False for _ in range(2 * n)]
    for u, v in itertools.chain(x.iterate_pairs(), y.iterate_pairs()):
        graph[u-1].append(v-1)
        graph[v-1].append(u-1)

    # BFS
    cycle_lens = []
    for start_u in range(2*n):
        if visited[start_u]:
            continue
        cycle_len = 1
        visited[start_u] = True
        q = deque()
        q.append(start_u)
        while len(q) > 0:
            u = q.popleft()
            for v in graph[u]:
                if not visited[v]:
                    q.append(v)
                    visited[v] = True
                    cycle_len += 1
        assert cycle_len % 2 == 0
        cycle_lens.append(cycle_len // 2)

    # Return
    assert sum(cycle_lens) == n
    return Partition(*cycle_lens)
