import os

operations_list = ['add', 'subtract', 'multiplication', 'division']

class var:
    def __init__(self, value, operation, ):
        self.value = value
        self.gradient = 0
        self.is_constant = False
        self.children = []
        self.operation = operation

    
    def __add__(self, other):
        self, other = check()

if __name__ == "__main__":