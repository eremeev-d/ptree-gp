from ptree_gp.primitives import Matching, Permutation


class PTree:
    def __init__(self, matching: Matching):
        self._matching = matching
        
        self._n = matching.get_number_of_pairs()
        self._children = dict()
        self._ancestor = {"R": None}
        
        pairs = [sorted(pair) for pair in matching.iterate_pairs()]
        ancestor = self._n + 2
        
        while len(pairs) > 0:
            pair = self._select_pair(pairs, ancestor)
            self._add_pair(pair, ancestor)
            pairs.remove(pair)
            ancestor += 1
        
    def _select_pair(self, pairs, ancestor):
        pairs = sorted(pairs)
        
        for pair in pairs:
            if pair[1] < ancestor:
                return pair
        raise RuntimeError("Something went wrong while constructing PTree")
        
        
    def _add_pair(self, pair, ancestor):
        if ancestor == ((2 * self._n) + 1):
                ancestor = "R"
                
#         print(pair, ancestor)
        self._ancestor[pair[0]] = ancestor
        self._ancestor[pair[1]] = ancestor
        self._children[ancestor] = sorted(pair)
        
    def _to_latex(self, node, no_inner_nums):
        if node == "R":
            s_node = node
        elif no_inner_nums and (node > self._n + 1):
            s_node = "$\\bullet$"
        else:
            s_node = str(node)
        
        s = "[" + s_node
        if node in self._children:
            for child in self._children[node]:
                s += " "
                s += self._to_latex(child, no_inner_nums)
        s += "]"
        return s
           
    def to_latex(self, no_inner_nums=True):
        return self._to_latex("R", no_inner_nums)


def get_transposition(n, i, j):
    perm = list(range(1, n+1))
    perm[i-1] = j
    perm[j-1] = i
    return Permutation(*perm)


def inspect_graph(matching, item_size=0.245):
    n = matching.get_number_of_pairs()
    
    neighbors = set()
    
    for i in range(1, 2*n+1):
        for j in range(i+1, 2*n+1):
#             if (i <= n+1) and (j <= n+1):
#                 continue
#             else:
            sigma = get_transposition(2*n, i, j)
            neighbor = sigma * matching
            if neighbor != matching:
                neighbors.add(neighbor)
    
    print("\\begin{figure}[ht]")
    
    print("\\begin{subfigure}[b]{" + str(item_size) + "\\textwidth}")
    print("\\begin{forest}")
    print(PTree(matching).to_latex())
    print("\\end{forest}")
    print("\\caption{" + str(list(matching.iterate_pairs())) + "}")
    print("\\end{subfigure}")
    
    for neighbor in sorted(neighbors, key = lambda m: m._value):
        print("\\begin{subfigure}[b]{" + str(item_size) + "\\textwidth}")
        print("\\begin{forest}")
        print(PTree(neighbor).to_latex())
        print("\\end{forest}")
        print("\\caption{" + str(list(neighbor.iterate_pairs())) + "}")
        print("\\end{subfigure}")
    
    print("\\caption{Matching=" + str(list(matching.iterate_pairs())) + "}")
    
    print("\\end{figure}")


def inspect_transposition(matching, i, j):
    n = matching.get_number_of_pairs()
    sigma = get_transposition(2 * n, i, j)
    
    print("\\begin{figure}")
    
    print("\\begin{subfigure}[b]{0.45\\textwidth}")
    print("\\begin{forest}")
    print(PTree(matching).to_latex())
    print("\\end{forest}")
    print("\\end{subfigure}")
    
    print("\\begin{subfigure}[b]{0.45\\textwidth}")
    print("\\begin{forest}")
    print(PTree(sigma * matching).to_latex())
    print("\\end{forest}")
    print("\\end{subfigure}")
    
    print("\\caption{Matching=" + str(list(matching.iterate_pairs())) + ". $\\sigma=(" + str(i) + ", " + str(j) + ")$}")
    
    print("\\end{figure}")


def main():
    matching = Matching((1, 2), (3, 4), (5, 6), (7, 8))
    inspect_graph(matching, item_size=0.245)

    matching = Matching((1, 2), (3, 4), (5, 16), (6, 7), (8, 14), (9, 10), (11, 12), (13, 15), (17, 18))
    inspect_transposition(matching, 11, 15)


if __name__ == "__main__":
    main()