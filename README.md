# MULTI-CLASS-CLASSIFICATION
## Aim:
To write a python program to implement the multi class classification algorithm .

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner / Google Colab

## Related Theoritical Concept:
Neural networks reflect the behaviour of the human brain. They allow programs to recognise patterns and solve common problems in machine learning. This is another option to either perform classification instead of logistics regression.

Classification is categorizing objects into groups. There are different types of classifications one of which is multiclass classification.In multi-class classification, the neural network has the same number of output nodes as the number of classes. Each output node belongs to some class and outputs a score for that class.

Class is a category for example Predicting animal class from an animal image is an example of multi-class classification, where each animal can belong to only one category.

Scores from the last layer are passed through a SoftMax layer. The SoftMax layer converts the score into probability values. At last, data is classified into a corresponding class, that has the highest probability value
![image](https://user-images.githubusercontent.com/78891098/168518341-3b5cbd46-23f7-464e-b8f8-287ec4b03c2d.png)



## Algorithm

    1.Import necessary libraries from packages.
    2.Assign x,y values from the given dataset by sklearn.
    3Count the number of key-value pairs in the datasets using counter libraries.
    4.Plot the x and y values in the chart using mathplotlib.pyplot.
    5.Label the values of x and y axis and add title to the graph.
    6.Save the file and execute the program.


## Program:

Program to implement the multi class classifier.
Developed by: mounika.s.c

RegisterNumber: 212219040084
```python
from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot
X, y = make_blobs(n_samples=1000, centers=3, random_state=1)
print(X.shape, y.shape)
counter = Counter(y)
print(counter)
for i in range(10):
    print(X[i], y[i])
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```


## Output:
![image](https://user-images.githubusercontent.com/78891098/168518588-78bb8046-e16c-4c44-ae64-324e3170a73b.png)



## Result:
Thus the python program to implement the multi class classification was implemented successfully.
