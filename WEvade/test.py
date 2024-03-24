import torch as t
from torch.autograd import Variable as V

def test(x):
    loss=x**3
    gx=t.autograd.grad(loss, x, create_graph=True)
    x1=gx[0]
    # print(x,gx,x1)
    return x1

x = t.Tensor([1])
x.requires_grad = True
x1=x

for i in range (2):
    print(i)
    x1=test(x1)
    print(x1)
    gx=t.autograd.grad(x1, x, create_graph=False)
    print(x1,gx)
# x2=test(x1)
# gx=t.autograd.grad(x2, x, create_graph=True)
# print(x2,gx)

# x3=test(x2)
# gx=t.autograd.grad(x3, x, create_graph=True)
# print(x3,gx)


    

loss=x1
gx=t.autograd.grad(loss, x, create_graph=False)
x_final=x+gx[0]
print(gx,x_final)