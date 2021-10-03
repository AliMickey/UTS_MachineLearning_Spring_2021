import numpy as np

class myDecisionTreeClassifer:
    # Method to construct a decision tree classifer
    def fit(self, input, output):
        data = input.copy()
        data[output.name] = output
        self.tree = self.decisionTree(data, data, input.columns, output.name)
        return self
        
    # Method to predict target class
    def predict(self, input):
        # Convert data into dictionary
        inputDict = input.to_dict(orient='records')
        predictions = []

        # Make and store prediction for each data point
        for sample in inputDict:
            predictions.append(self.makePrediction(sample, self.tree, 1.0))
        return predictions

    # Method to calculate entropy
    def entropy(self, attributeColumn):
        # Calculate frequency of unique values within data
        values, counts = np.unique(attributeColumn, return_counts=True)
        entropyList = []

        for i in range(len(values)):
            probability = counts[i]/np.sum(counts)
            entropyList.append(-probability*np.log2(probability))

        entropyTotal = np.sum(entropyList)

        # Return sum of entropy list 
        return entropyTotal

    # Method to calculate the information gain
    def informationGain(self, data, featureAttrName, targetAttrName):
        # Get sum of all entropy values for targets 
        entropyTotal = self.entropy(data[targetAttrName])

        # Calculate frequency of unique values within data
        values, counts = np.unique(data[featureAttrName], return_counts=True)
        weightedEntropyList = []

        for i in range(len(values)):
            subset_probability = counts[i]/np.sum(counts)
            subset_entropy = self.entropy(data.where(data[featureAttrName]==values[i]).dropna()[targetAttrName])
            weightedEntropyList.append(subset_probability*subset_entropy)

        totalWeightedEntropy = np.sum(weightedEntropyList)

        # Return information gain
        informationGain = entropyTotal - totalWeightedEntropy

        return informationGain

    # Method to make decision tree
    def decisionTree(self, data, originalData, featureAttrNames, targetAttrName, parentNodeClass=None):
        uniqueClasses = np.unique(data[targetAttrName])
        # If every element belongs to the same class, return all elements
        if len(uniqueClasses) <= 1:
            return uniqueClasses[0]
        # If there are no more attributes but data is spread across different classes, return data containing the majority of one class
        elif len(data) == 0:
            majorityClassIndex = np.argmax(np.unique(originalData[targetAttrName], return_counts=True)[1])
            return np.unique(originalData[targetAttrName])[majorityClassIndex]
        # If there are no more attributes left to split, return parent node.
        elif len(featureAttrNames) == 0:
            return parentNodeClass
        # Else make a new branch
        else:
            # Identify parent class of current branch
            majorityClassIndex = np.argmax(np.unique(data[targetAttrName], return_counts=True)[1])
            parentNodeClass = uniqueClasses[majorityClassIndex]

        # Calculate information gain and select highest value to split the data for each attribute
        ig_values = [self.informationGain(data, feature, targetAttrName) for feature in featureAttrNames]
        bestFeatureIndex = np.argmax(ig_values)
        bestFeature = featureAttrNames[bestFeatureIndex]

        # Make empty tree structure
        tree = {bestFeature: {}}

        # Make best attribute the parent node and remove
        featureAttrNames = [i for i in featureAttrNames if i != bestFeature]

        # Construct child nodes
        parentAttributeValues = np.unique(data[bestFeature])
        for value in parentAttributeValues:
            sub_data = data.where(data[bestFeature] == value).dropna()

            # Recursive loop call
            subtree = self.decisionTree(sub_data, originalData, featureAttrNames, targetAttrName, parentNodeClass)

            # Add all child nodes to main tree
            tree[bestFeature][value] = subtree

        return tree

    # Method to run prediction
    def makePrediction(self, sample, tree, default=1):
        # Map data sample to tree
        for attribute in list(sample.keys()):
            if attribute in list(tree.keys()):
                # Return result for feature if it exists in tree
                try:
                    result = tree[attribute][sample[attribute]]
                except:
                    return default

                result = tree[attribute][sample[attribute]]

                # Recursively run predicter until there are no more attributes left 
                if isinstance(result, dict):
                    return self.makePrediction(sample, result)
                else:
                    return result