import numpy as as np
class KDNode:
    def __init__(self, val):
        self.val = val
    
    def setLeft(left):
        self.left = left

    def setRight(right):
        self.right = right

class KDTree:
    """
       @param Numpy.Matrix dataset
    """
    def __init__(self, dataset):
        self.root = self.buildTree(dataset, 0)
        
    def dist(x,y):   
        return np.sqrt(np.sum((x-y)**2))

    def buildTree(dataset, depth):
        m = dataset.shape[0]
        n = dataset.shape[1]

        axis = depth % n
        col = dataset[:, axis].A1
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
            row = dataset[i, :].getA1().tolist()
            if (row[col] < median):
                left_list = left_list.append(row)
                #node.setLeft(buildTree(new_dataset, depth+1))
            else:
                right_list = right_list.append(row)
        if not left_lsit:
            left_dataset = np.matrix(left_list)
            node.setLeft(buildTree(left_dataset, depth+1))
        else:
            node.setLeft(None)

        if not right_list:
            right_dataset = np.matrix(right_list)
            node.setRight(buildTree(right_dataset, depth+1))
        else:
              node.setRight(None)
        return node

    def search(node, x, depth):
        if (node.left == None && node.right == None):
            nearest = node
            return nearest
        else:
            axis = depth % n
            if (x[axis] < node.val[axis] && node.left != None):
                nearest = search(node.left, x, depth+1):
                if (self.dist(node.val, x) < self.dist(nearest.val, x)):
                    nearest = node
                if (node.right != None && self.dist(node.right.val, x) <= self.dist(nearest.val, x)):
                    nearest = search(node.right, x, depth+1)
                return nearest
            else if (x[axis] >= node.val[axis] && node.right != None):
                nearest = search(node.right, x, depth+1)
                if (self.dist(node.val, x) < self.dist(nearest.val, x)):
                    nearest = node
                if (node.left != None && self.dist(node.left.val, x) <= self.dist(nearest.val, x)):
                    nearest = search(node.right, x, depth+1)
                return nearest