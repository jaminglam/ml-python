import numpy as np
class KDNode:
    def __init__(self, val):
        self.val = val
    
    def setLeft(self, left):
        self.left = left

    def setRight(self, right):
        self.right = right

class KDTree:
    """
       @param Numpy.Matrix dataset
    """
    def __init__(self, dataset):
        self.root = self.buildTree(dataset, 0)
        
    def dist(self, x,y):   
        return np.sqrt(np.sum((x-y)**2))

    def buildTree(self, dataset, depth):
        if dataset is None:
            return None
        m = dataset.shape[0]
        n = dataset.shape[1]

        axis = depth % n
        col = dataset[:, axis].A1
        #print "col"
        #print col
        sorted_col = np.sort(col)
        median = sorted_col[sorted_col.size / 2]
        for i in range(0, m):
            if (col[i] == median):
                domain_row_index = i
                break
        node_val = dataset[domain_row_index, :].A1
        node = KDNode(node_val)
        left_list = []
        right_list = []
        for i in range(0, m):
            if (i == domain_row_index):
                continue
            row = dataset[i, :].getA1().tolist()
            if (row[axis] < median):
                left_list.append(row)
            else:
                right_list.append(row)
        if len(left_list) != 0:
            left_dataset = np.matrix(left_list)
        else:
            left_dataset = None
        if len(right_list) != 0:
            right_dataset = np.matrix(right_list)
        else:
            right_dataset = None
        node.setLeft(self.buildTree(left_dataset, depth+1))
        node.setRight(self.buildTree(right_dataset, depth+1))
        return node

    def searchNearest(self, node, x, depth):
        if (node.left == None and node.right == None):
            nearest = node
            return nearest
        else:
            axis = depth % n
            if (x[axis] < node.val[axis] and node.left != None):
                nearest = self.searchNearest(node.left, x, depth+1)
                if (self.dist(node.val, x) < self.dist(nearest.val, x)):
                    nearest = node
                if (node.right != None and self.dist(node.right.val, x) <= self.dist(nearest.val, x)):
                    nearest = self.searchNearest(node.right, x, depth+1)
                return nearest
            elif (x[axis] >= node.val[axis] and node.right != None):
                nearest = self.searchNearest(node.right, x, depth+1)
                if (self.dist(node.val, x) < self.dist(nearest.val, x)):
                    nearest = node
                if (node.left != None and self.dist(node.left.val, x) <= self.dist(nearest.val, x)):
                    nearest = self.searchNearest(node.right, x, depth+1)
                return nearest

    """
       traverse is a pre-order traversal
    """
    def traverse(self, node, visited):
        if (node == None):
            visited.append(None)
            return visited
        else:
            visited.append(node.val)
            visited = self.traverse(node.left, visited)
            visited = self.traverse(node.right, visited)
            return visited

if __name__ == "__main__":
    dataset = np.matrix([[2.,3.],[5.,4.],[9.,6.],[4.,7.],[8.,1.],[7.,2.]])
    tree = KDTree(dataset)

    visited = tree.traverse(tree.root, [])
    print visited