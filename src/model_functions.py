from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns




def lrm(X, y, rs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs)
    
    # Instantiate the model
    lr = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')

    # Fit the model
    lr.fit(X_train, y_train)
    
    # Generate predictions
    y_hat_train = lr.predict(X_train)
    y_hat_test = lr.predict(X_test)
    y_hat_test_proba = lr.predict_proba(X_test)
    return y_test, y_hat_test, y_hat_test_proba
    
def show_cm(cnf_matrix):
    plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) 

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.ylabel('True Position')
    plt.xlabel('Predicted Position')

    class_names = set(y) 
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cnf_matrix.max() / 2.  
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],
                     horizontalalignment='center',
                     color='white' if cnf_matrix[i, j] > thresh else 'black')
    plt.colorbar()
    plt.show()
    