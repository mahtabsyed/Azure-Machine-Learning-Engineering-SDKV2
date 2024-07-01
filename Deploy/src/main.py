import os
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score


# define functions
def main(args):
    # enable auto logging
    current_run = mlflow.start_run()
    mlflow.sklearn.autolog(log_models=False)

    # read in data
    df = pd.read_csv(args.titanic_csv)
    model = model_train('Survived', df, args.randomstate)
    mlflow.end_run()

def model_train(LABEL, df, randomstate):
    print('df.columns = ')
    print(df.columns)
    
    df['Embarked'] = df['Embarked'].astype(object)
    df['Loc'] = df['Loc'].astype(object)
    df['Loc'] = df['Sex'].astype(object)
    df['Pclass'] = df['Pclass'].astype(float)
    df['Age'] = df['Age'].astype(float)
    df['Fare'] = df['Fare'].astype(float)
    df['GroupSize'] = df['GroupSize'].astype(float)

    y_raw           = df[LABEL]
    columns_to_keep = ['Embarked', 'Loc', 'Sex','Pclass', 'Age', 'Fare', 'GroupSize']
    X_raw           = df[columns_to_keep]

    print(X_raw.columns)
     # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=randomstate)
    
    #use Logistic Regression estimator from scikit learn
    lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    preprocessor = buildpreprocessorpipeline(X_train)
    
    #estimator instance
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lg)], verbose=True)

    model = clf.fit(X_train, y_train)
    
    print('type of X_test = ' + str(type(X_test)))
          
    y_pred = model.predict(X_test)
    
    print('*****X_test************')
    print(X_test)
    
    #get the active run.
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))

    acc = model.score(X_test, y_test )
    print('Accuracy:', acc)
    MlflowClient().log_metric(run.info.run_id, "test_acc", acc)
    
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' , auc)
    MlflowClient().log_metric(run.info.run_id, "test_auc", auc)
    
    
    # Signature
    signature = infer_signature(X_test, y_test)

    # Conda environment
    custom_env =_mlflow_conda_env(
        additional_conda_deps=["scikit-learn==1.1.3"],
        additional_pip_deps=["mlflow<=1.30.0"],
        additional_conda_channels=None,
    )

    # Sample
    input_example = X_train.sample(n=1)

    # Log the model manually
    mlflow.sklearn.log_model(model, 
                             artifact_path="model", 
                             conda_env=custom_env,
                             signature=signature,
                             input_example=input_example)


    
    return model



def buildpreprocessorpipeline(X_raw):

    categorical_features = X_raw.select_dtypes(include=['object', 'bool']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    #categorical_features = ['Sex', 'Embarked', 'Loc']
    categorical_transformer = Pipeline(steps=[('onehotencoder', 
                                               OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])


    #numeric_features = ['Pclass', 'Age', 'Fare', 'GroupSize']    
    numeric_transformer1 = Pipeline(steps=[('scaler1', SimpleImputer(missing_values=np.nan, strategy = 'mean'))])
    

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric1', numeric_transformer1, numeric_features),
            ('categorical', categorical_transformer, categorical_features)], remainder='drop')
    
    return preprocessor



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--titanic-csv", type=str)
    parser.add_argument("--randomstate", type=int, default=42)

    # parse args
    args = parser.parse_args()
    print(args)
    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
