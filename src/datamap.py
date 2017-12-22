from pandas import Series,DataFrame
import numpy as np

class DataMap:
    # the list of original classes
    classes = []
    map = None
    # the index of classes
    index = None
    class_num = 0
    # the offset of index.(default from 0)
    offset = 0

    def __init__(self,classes, offset = 0):
        self.classes = classes
        self.class_num = len(classes)
        self.offset = offset
        self.index = np.arange(0,self.class_num) + self.offset
        self.map = Series(self.index,index=classes)

    def get_index(self):
        return self.index

    def get_class(self):
        return self.classes

    def class2index(self, data):
        """
        index the data
        """
        num = len(data)
        res = np.zeros(num, dtype=int)
        for i in range(0,num):
            res[i] = self.map[data[i]]

        return res

    def index2class(self,idx):
        """
        covert index to its corresponding class
        """
        num = len(idx)
        idx = np.array(idx) - self.offset
        res = num * [None]
        keys = self.map.keys().tolist()
        for i in range(0,num):
            res[i] = keys[idx[i]]

        return res

    def class2matrix(self,data):
        """
        convert a list of classes to matrix expression
        """
        idx = self.class2index(data)
        res = self.index2matrix(idx)
        return res

    def matrix2class(self,matrix):
        """
        convert a matrix expression to its original classes
        """
        idx = matrix2index(matrix)
        return self.index2class(idx)

    def matrix2index(self,matrix):
        idx = np.argmax(matrix,axis=0)
        idx += self.offset
        idx = idx.astype(int)

        return idx

    def index2matrix(self,idx):
        """
        use vector to express each y[i]
        """
        num = np.size(idx)
        idx = idx - self.offset
        idx = idx.astype(int)
        eye = np.eye(self.class_num)
        res = np.zeros((self.class_num, num), dtype=int)
        for i in np.arange(0,num):
            res[:,i] = eye[:,idx[i]]

        return res

if __name__ == '__main__':
    map = DataMap(['one','two','three','four'],offset=1)
    print (map.get_index())
    print (map.class2index(['one','two','three','four']))
    print (map.index2class([1,2,3,4]))
    print (map.class2matrix(['one','two','three','four']))
    print (map.matrix2class(np.eye(4)))
