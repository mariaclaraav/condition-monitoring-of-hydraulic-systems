import joblib
import warnings
import time
import os
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ModelExperiment:
    """
    A class to conduct experiments with various machine learning models on a given dataset.
    
    Attributes:
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_test (array-like): Testing feature data.
        y_test (array-like): Testing target data.
        models (dict): A dictionary of initialized machine learning models.
        parameters (dict): A dictionary of hyperparameters for grid search for certain models.
        only_execute (list): A list of models to execute without hyperparameter tuning.
        search_execute (list): A list of models to execute with hyperparameter tuning using grid search.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, only_execute=None, search_execute=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = self._initialize_models()
        self.parameters = self._initialize_parameters()
        self.only_execute = only_execute if only_execute is not None else ['Logistic Regression', 'Linear Discriminant', 'Gaussian Process', 'Naive Bayes']
        self.search_execute = search_execute if search_execute is not None else ['Linear SVM', 'Decision Tree', 'Random Forest', 'Neural Net', 'AdaBoost']
        
    def _initialize_models(self):
        """
        Initializes the machine learning models to be used in the experiment.
       
        """
        return {            
            'Logistic Regression': LogisticRegression(),
            'Linear Discriminant': LinearDiscriminantAnalysis(),
            'Nearest Neighbors': KNeighborsClassifier(),
            'Linear SVM': SVC(kernel='linear', gamma='auto'),
            'Gaussian Process': GaussianProcessClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Neural Net': MLPClassifier(alpha=1, max_iter=1000),
            'AdaBoost': AdaBoostClassifier(algorithm='SAMME'),
            'Naive Bayes': GaussianNB()
        }
        
    def _initialize_parameters(self):
        """
        Initializes the hyperparameters for grid search for certain models.
    
        """
        return {
            'Nearest Neighbors': {'n_neighbors': [1, 3, 5, 7]},
            'Linear SVM': {'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]},
            'Decision Tree': {'max_depth': [None, 3, 5, 7]},
            'Random Forest': {'n_estimators': [30, 100, 300]},
            'Neural Net': {'hidden_layer_sizes': [30, 100, 300], 'activation': ['logistic', 'tanh', 'relu']},
            'AdaBoost': {'n_estimators': [30, 100, 300]}
        }
        
    def run_experiments(self, save_dir):
        """
        Runs the experiments with the initialized models, both with and without hyperparameter tuning.
        
        """
        ans = {}
        best_score = 0
        best_model = None
        best_model_name = ''
        print('\nStarting to run the models..\n')
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
                
        # Execute simple models
        for c in self.only_execute:
            start = time.process_time()
            pipeline = Pipeline([
                ('transformer', StandardScaler()), 
                ('estimator', self.models[c])
            ])
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            ans[c] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
            print(f'Elapsed time of {c} is {time.process_time() - start:.6f} seconds.')
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1}')
            print('-'*40)
                      
            best_model = pipeline
            best_model_name = ''.join([word[0] for word in c.split()])  # Get initials of the model name
                
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
                
            # Save the best model
            save_path = os.path.join(save_dir, f'{best_model_name}.pkl')
            joblib.dump(best_model, save_path)
            print(f'Best model saved: {save_path}')
        
                
        # Execute models with grid search
        for c in self.search_execute:
            start = time.process_time()
            clf = GridSearchCV(self.models[c], param_grid=self.parameters[c])
            pipeline = Pipeline([
                ('transformer', StandardScaler()), 
                ('estimator', clf)
            ])
            pipeline.fit(self.X_train, self.y_train)
            best_estimator = clf.best_estimator_
            y_pred = pipeline.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            ans[c] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
            print(f'Elapsed time of {c} is {time.process_time() - start:.6f} seconds.')
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1}')
            print(best_estimator)
            print('-'*40)
            
           
            
            best_model = pipeline
            best_model_name = ''.join([word[0] for word in c.split()])  # Get initials of the model name
                
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
                
            # Save the best model
            save_path = os.path.join(save_dir, f'{best_model_name}.pkl')
            joblib.dump(best_model, save_path)
            print(f'Best model saved: {save_path}')
        
        return ans
    
    
class ModelEvaluator:
    """
    A class to load, evaluate, and visualize machine learning models stored in a directory.

    Attributes
    ----------
    models_dir : str
        The directory containing the saved models.
    X_test : DataFrame
        The test data features.
    y_test : Series
        The true labels for the test data.
    models : dict
        A dictionary where keys are model names and values are the loaded model objects.
    friendly_names : dict
        A dictionary where keys are model short names and values are friendly names.
    """

    def __init__(self, models_dir, X_test, y_test, output_dir):
        self.models_dir = models_dir
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir
        self.models = self.load_models()
        self.friendly_names = {
            'A': 'AdaBoost',
            'NN': 'Neural Network',
            'RF': 'Random Forest',
            'DT': 'Decision Trees',
            'LR': 'Logistic Regression',
            'NB': 'Naive Bayes'
        }

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_models(self):
        """
        Loads all models from the specified directory.
        """
        models = {}
        for file_name in os.listdir(self.models_dir):
            if file_name.endswith('.pkl'):
                model_path = os.path.join(self.models_dir, file_name)
                model_name = file_name.split('.')[0]
                models[model_name] = joblib.load(model_path)
                print(f'Model loaded: {model_name}')
        return models

    def evaluate_models(self):
        """
        Evaluates each model on the test data and prints performance metrics.
        """
        results = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            self.plot_confusion_matrix(y_pred, model_name)
        return results

    def plot_confusion_matrix(self, y_pred, model_name):
        """
        Plots the confusion matrix for a given model's predictions.
        """
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        friendly_name = self.friendly_names.get(model_name, model_name)
        plt.title(f'Confusion Matrix for {friendly_name}')
        
        # Save the plot to the output directory
        output_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(output_path)
        plt.close()