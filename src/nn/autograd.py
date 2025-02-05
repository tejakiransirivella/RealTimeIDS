class Tensor:
    def __init__(self, value, requires_grad=False):
        self.value = value
        self.requires_grad = requires_grad
        self.grad = 0.0 if requires_grad else None
        self._backward = lambda: None

    def backward(self, grad=1.0):
        # Seed the initial gradient of the final tensor
        self.grad = grad
        self._backward()

def multiply(tensor_x, tensor_y):
    z = Tensor(tensor_x.value * tensor_y.value, requires_grad=True)

    def _backward():
        if tensor_x.requires_grad:
            tensor_x.grad += tensor_y.value * z.grad 
            
        if tensor_y.requires_grad:
            tensor_y.grad += tensor_x.value * z.grad      
        
        tensor_x._backward()
        tensor_y._backward()

    z._backward = _backward
    return z

def add(tensor_z, tensor_y):
    z = Tensor(tensor_z.value + tensor_y.value, requires_grad=True)

    def _backward():
        if tensor_z.requires_grad:
            tensor_z.grad += z.grad
        if tensor_y.requires_grad:
            tensor_y.grad += z.grad
        tensor_z._backward()
        tensor_y._backward()

    z._backward = _backward
    return z

# Example usage
x = Tensor(3.0, requires_grad=True)
y = Tensor(4.0, requires_grad=True)

z_mul = multiply(x, y)  # z_mul = x * y = 3 * 4
z = add(z_mul, y)       # z = z_mul + y = 12 + 4
                        # z= x*y + y
z.backward()

print(f"Gradient w.r.t x: {x.grad}")  # Expected: y = 4.0
print(f"Gradient w.r.t y: {y.grad}")  # Expected: x + 1 = 3.0 + 1.0 = 4.0
