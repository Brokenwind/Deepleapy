from pandas import Series,DataFrame
import numpy as np

class DataMap:
    # a array of original classes
    classes = None
    map = None
    # the index of classes
    index = None
    class_num = 0
    # the offset of index.(default from 0)
    offset = 0

    def __init__(self,classes, offset = 0):
        self.classes = np.array(classes).flatten()
        self.class_num = self.classes.size
        self.offset = offset
        self.index = np.arange(0,self.class_num) + self.offset
        self.map = Series(self.index,index=classes)

    def get_index(self):
        return self.index

    def get_class(self):
        return self.classes

    def class2index(self, data):
        """
        Tranform a class array to its index array
        Parameters
        ----------
        data : list/tuple/np.ndarray
            a array/tuple/array of class names
        Returns
        -------
        res : np.ndarray
            a array of transformed indexes
        """
        data = np.array(data)
        num = data.size
        res = np.zeros(num, dtype=int)
        for i in range(0,num):
            res[i] = self.map[data[i]]

        return res

    def index2class(self,idx):
        """
        Transform index to its corresponding class
        Parameters
        idx : list/tuple/np.ndarray
            a array/tuple/array of indexes
        ----------
        Returns
        -------
        res : np.ndarray
            a list of transformed class names
        """
        idx = np.array(idx).flatten()
        num = len(idx)
        idx = np.array(idx) - self.offset
        res = num * [None]
        keys = self.map.keys().tolist()
        for i in range(0,num):
            res[i] = keys[idx[i]]

        return res

    def class2matrix(self,data):
        """
        Transform a list of class names to matrix expression
        It is used when class number bigger than 2
        Parameters
        ----------
        data : list/tuple/np.ndarray
            a array/tuple/array of class names
        Returns
        -------
        res : 2D np.ndarray(classes,samples)
            a 2D array of transformed matrix
        """
        idx = self.class2index(data)
        res = self.index2matrix(idx)
        return res

    def matrix2class(self,matrix):
        """
        Transform a matrix expression to a list of class names
        Parameters
        ----------
        matrix : 2D np.ndarray(classes,samples)
            a 2D array of transformed matrix
        Returns
        -------
            a list of class names
        """
        idx = self.matrix2index(matrix)
        return self.index2class(idx)

    def index2matrix(self,idx):
        """
        Transform a list/tuple/array of indexes to its matrix expression
        Parameters
        ----------
        matrix : list/tuple/np.ndarray
            a array/tuple/array of indexes
        Returns
        -------
        res : 2D np.ndarray (classes,samples)
            a 2D array of transformed matrix
        """
        idx = np.array(idx)
        idx = idx.flatten()
        if self.class_num == 2:
            idx = idx.reshape((1,idx.size))
            if len(np.unique(idx)) != 2:
                raise ValueError('The number of different idx is not consistent with the number of different class name')
            return idx
        num = np.size(idx)
        idx = idx - self.offset
        idx = idx.astype(int)
        eye = np.eye(self.class_num)
        res = np.zeros((self.class_num, num), dtype=int)
        for i in np.arange(0,num):
            res[:,i] = eye[:,idx[i]]

        return res

    def matrix2index(self,matrix):
        """
        Transform matrix expression of label to a indexes array
        Parameters
        ----------
        matrix : 2D np.ndarray (classes,samples)
            a 2D array of transformed matrix
        Returns
        -------
        idx : np.ndarray
            a array of indexes
        """
        matrix = np.array(matrix)
        if matrix.ndim == 1 or matrix.shape[0] == 1 or matrix.shape[0] == 1:
            matrix[matrix >= 0.5] = 1
            matrix[matrix <  0.5] = 0
            idx = matrix.astype(int)
        else:
            idx = np.argmax(matrix,axis=0)
            idx += self.offset
            idx = idx.astype(int)

        return idx

    def matrix2matrix(self,matrix):
        """
        Transform a probility matrix to a matrix with 0 and 1
        """
        idx = self.matrix2index(matrix)
        return self.index2matrix(idx)

if __name__ == '__main__':
    map = DataMap(['one','two','three','four'],offset=1)
    print (map.get_index())
    print (map.class2index(['one','two','three','four']))
    print (map.index2class([1,2,3,4]))
    print (map.class2matrix(['one','two','three','four']))
    print (map.matrix2class(np.eye(4)))

    map = DataMap(['zero','one'])
    print (map.index2matrix([0,1,0,1]))
    print (map.matrix2index(np.array([[0,1,0,1]])))
