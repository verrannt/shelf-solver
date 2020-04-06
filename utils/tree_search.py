class Tree(object):
    def __init__(self, datatype):
        self.data = []
        if datatype == "stack":
            self.op = -1
        elif datatype == "queue":
            self.op = 0
        else:
            raise ValueError("Datatype needs to be either stack or queue.")
        
    def __call__(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def add(self, item):
        self.data.append(item)
        
    def get(self):
        return self.data.pop(self.op)
    
    def is_not_empty(self):
        return bool(self.data)

class Queue(object):
    def __init__(self):
        self.data = []
         
    def __call__(self):
        return self.data
  
    def __len__(self):
        return len(self.data)
    
    def enqueue(self, item):
        self.data.append(item)
        
    def dequeue(self):
        return self.data.pop(0)

    def is_not_empty(self):
        return bool(self.data)
    
class Stack(object):
    def __init__(self):
        self.data = []
        
    def __call__(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def push(self, item):
        self.data.append(item)
        
    def pop(self):
        return self.data.pop(-1)
    
    def is_not_empty(self):
        return bool(self.data)
    
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        if self.parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0