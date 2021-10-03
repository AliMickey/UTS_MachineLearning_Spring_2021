from sys import argv
from graphviz import Source
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from myClassifer import myDecisionTreeClassifer

# Return iris dataset formatted as data frame
def irisSetup(simple=False):
    iris = load_iris(as_frame=True)
    if simple == "True":
        # Simplify by using only first two attributes and two species for binary classification
        irisData = iris['data'][['sepal length (cm)', 'sepal width (cm)']][0:100]
        irisTarget = iris['target'][0:100]
        featureNames = iris['feature_names'][0:2]
        targetNames = iris['target_names'][0:2]
    elif simple == "False":
        irisData = iris['data']
        irisTarget = iris['target']
        featureNames = iris['feature_names']
        targetNames = iris['target_names']

    return [irisData, irisTarget, featureNames, targetNames]
        
# Sklearn's 'ready to go' decision tree classifier - FOR COMPARISON ONLY
def toolboxClassifer(iris):
    print("Running sklearn classifer")

    # Map iris dataset to variables
    irisData, irisTarget, featureNames, targetNames = iris

    # Split into test and training as 0.3/0.7
    dataTrain, dataTest, targetTrain, targetTest = train_test_split(irisData, irisTarget, random_state=50, test_size=0.30)

    # Setup and run model
    model = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=25)
    model.fit(dataTrain, targetTrain)

    # Calculate accuracy
    print('Accuracy Score on train data: ', round(accuracy_score(y_true=targetTrain, y_pred=model.predict(dataTrain)),3))
    print('Accuracy Score on the test data: ', round(accuracy_score(y_true=targetTest, y_pred=model.predict(dataTest)),3))

    # Export tree visualisation
    #exportImage(model, featureNames, targetNames)

# Makes instance of my classifer
def myClassifer(iris):
    print("Running myClassifer")

    # Map iris dataset to variables
    irisData, irisTarget, featureNames, targetNames = iris

    # Split into test and training as 0.3/0.7
    dataTrain, dataTest, targetTrain, targetTest = train_test_split(irisData, irisTarget, random_state=50, test_size=0.30)

    # Setup and run model
    model = myDecisionTreeClassifer()
    model.fit(dataTrain, targetTrain)
    
    # Calculate accuracy
    print('Accuracy Score on train data: ', round(accuracy_score(y_true=targetTrain, y_pred=model.predict(dataTrain)),3))
    print('Accuracy Score on the test data: ', round(accuracy_score(y_true=targetTest, y_pred=model.predict(dataTest)),3))

# Visualise tree
def exportImage(model, featureNames, targetNames):
    graph = Source(export_graphviz(model, out_file=None, feature_names=featureNames, class_names=targetNames, filled=True))
    graph.format="png"
    graph.render('tree', view=True)
    print("Tree visualisation exported to tree.png")


# Pass through True for the simplified Iris Flower dataset
arg = argv[1]
print(f"Running classifers with simple as {arg}")
print("---------------------------------------")
toolboxClassifer(irisSetup(arg))
print('\n')
myClassifer(irisSetup(arg))