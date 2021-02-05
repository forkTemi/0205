import numpy as np
import pandas as pd

eps = np.finfo(float).eps

from numpy import log2 as log

def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    valures = df[Class].unique()
    for value in valures:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += fraction*np.log2(fraction)
    return entropy

def find_entropy_attribute(df,attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df[Class]==target_variable])
            den = len(df[attribute][df[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
            fraction2 = den/len(df)
            entropy2 +=-fraction2*entropy
    return abs(entropy2)

def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]

def get_subtable(df,node,value):
    return df[df[node]==value].reset_index(drop=True)

def buildTree(df,tree=None):
    Class = df.keys()[:-1]
    node = find_winner(df)
    attValue = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}
    for value in attValue:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['policy'],return_counts=True)
        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)
    return tree

def predict(inst,tree):
    for node in tree.keys():
        value = inst[node]
        tree = tree[node][value]
        prediction = 0
        if type(tree) is dict:
            prediction = predict(inst,tree)
        else:
            prediction = tree
            break;
    return prediction

dataset = {'usage':['<90%','<90%','<90%','>=90%','>=90%','>=90%','>90%'],
           'archfree':['Yes','No','Yes','Yes','No','No','Yes'],
           'datafree':['Yes','No','No','Yes','Yes','No','No'],
           'policy':['NoAction','NoAction','NoAction','Increase-Fra','Migrate_To_DATADG','NoAction','Increase-Fra']
           }

testset = {'usage':['>=90%'],
           'archfree':['Yes'],
           'datafree':['No'],
           'action':['Increase-Fra']
           }

df = pd.DataFrame(dataset,columns=['usage','archfree','datafree','policy'])
ts = pd.DataFrame(testset,columns=['usage','archfree','datafree','action'])


print("训练集:")
print(df)
inst = ts.iloc[0]
tree = buildTree(df)
import pprint
print("决策树:")
pprint.pprint(tree)
print("测试集:")
print(inst)
prediction = predict(inst,tree)
print("将采取策略:",prediction)