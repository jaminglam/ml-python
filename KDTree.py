import numpy as np

class KDNode:
    def __init__(self, val):
        self.val = val
    
    def setLeft(self, left):
        self.left = left

    def setRight(self, right):
        self.right = right

    def setAxis(self, axis):
        self.axis = axis

class KDTree:
    """
       @param Numpy.Matrix dataset
    """
    def __init__(self, dataset):
        self.root = self.buildTree(dataset, 0)
        
    def dist(self, x, y):   
        return np.sqrt(np.sum((x-y)**2))

    def node_dist(self, node1, node2):
        return self.dist(node1.val, node2.val)

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
        node.setAxis(axis)
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

    def nn(self, node, x_node, depth, nearest, min_dist):
        if node == None:
            return None
        if (node != None and node.left == None and node.right == None):
            dist = self.node_dist(x_node, node)
            if (dist <= min_dist):
                return node, dist
            else:
                return nearest, min_dist
        cur_dist = self.node_dist(node, x_node)
        if (x_node.val[node.axis] < node.val[node.axis]):
            first = "left"
            first_node, sec_node = node.left, node.right
        else:
            first = "right"
            first_node, sec_node = node.right, node.left
        if (cur_dist <= min_dist):
            nearest = node
            min_dist = cur_dist
        if (first_node != None):
            new_nearest, new_min_dist = self.nn(first_node, x_node, depth+1, nearest, min_dist)
            if (new_min_dist <= min_dist):
                nearest = new_nearest
                min_dist = new_min_dist
        if (first == "left" and x_node.val[node.axis] + min_dist >= node.val[node.axis] and sec_node != None):
            new_nearest, new_min_dist = self.nn(sec_node, x_node, depth+1, nearest, min_dist)
            if (new_min_dist <= min_dist):
                nearest = new_nearest
                min_dist = new_min_dist
        elif (first == "right" and x_node.val[node.axis] - min_dist < node.val[node.axis]):
            new_nearest, new_min_dist = self.nn(sec_node, x_node, depth+1, nearest, min_dist)
            if (new_min_dist <= min_dist):
                nearest = new_nearest
                min_dist = new_min_dist
        return nearest, min_dist

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

    #visited = tree.traverse(tree.root, [])
    #print visited
    x = [8.,2.]
    x_node = KDNode(x)
    nearest, min_dist = tree.nn(tree.root, x_node, 0, tree.root, tree.node_dist(tree.root, x_node))
    print ("nearest %s min_dist %d" %(nearest.val, min_dist))