'''
Script to unify evaluation proceedure across all models
author: Harry Coppock
Qs to: harry.coppock@imperial.ac.uk
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import json
import seaborn as sns
import geopandas
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            recall_score,
                            precision_score,
                            make_scorer,
                            RocCurveDisplay,
                            plot_roc_curve,
                            PrecisionRecallDisplay,
                            roc_curve,
                            auc,
                            roc_auc_score,
                            precision_recall_curve,
                            average_precision_score)


class EvalMetrics():
    '''
    Class representing all the metrics which we want in this
    COVID project
    Attributes:
    -----------
    predictions: pd.Dataframe
        4 x columns
            barcode: corresponding barcode ids
            labels: ground truth {'Positive', 'Negative'}
            preds: the model prediction {'Positive', 'Negative'}
            logits: float probabilty of COVID-19 [0,1]
            
    meta_data: pd:Dataframe
        na colums - the full study data meta data
    location: str
        absolute path to save the result metrics
    split: str
        whether this is eval on the devel or test splits {'development', 'test'}
    matched: bool
        whether to perform a version of matching/blocking
    type_analysis:
        is this standard analysis or are you running meta analysis. This is just
        for saving a logging the different metrics.
    description: str
        short description of run - used for the title in the plots
    symptoms: nested list
        a list of combination of symptoms you want to control for in the meta 
        analysis. See DEFAULT_SYMPTOMS for an example
    geos: nested list
        a list of combinations of geographical locations to control for.
    classifier: sklearn trained classifier
        This is ONLY needed when the classifier is required to calculate the
        fpr and tpr. For NNs simply return the logits
    '''
    
    DEFAULT_SYMPTOMS=[
            ['Cough (any)'], 
            ['No symptoms'], 
            ['Cough (any)', 'Fatigue', 'Headache']] 
    
    DEFAULT_GEO= [['Cornwall'], ['Mendip', 'South Somerset'], ['Leeds']]

    def __init__(self,
                predictions,
                meta_data,
                location,
                split,
                symptoms=DEFAULT_SYMPTOMS,
                geos=DEFAULT_GEO,
                description=None,
                matched=False,
                classifier=None):

        self.predictions = predictions
        self.meta_data = meta_data
        self.location = location
        self.metrics = {}
        self.split = split
        self.matched = matched
        self.symptoms = symptoms
        self.type_analysis = 'standard'
        self.description = self.location if description == None else description
        self.merged = self.merge_meta_predictions()
        self.load_geo_data()
        self.classifier = classifier
        if self.classifier != None:
            
            print("You need the trainX and trainY along with the SVM trained" \
                    "model if you want to calculate FPR and TPR therefore please" \
                    "calculate roc and pr curve in main script") 
        ### run some checks ###
        #self.check_location_path()
        #self.check_labels_consistent()
        ### standard metrics ###
        self.genMetrics(self.merged)
        ### meta analysis ###
        #symptoms:
        for symptom_combo in symptoms:
            self.control_symptom_analysis(symptom_combo)
        for geo in geos:
            self.control_geo_analysis(geo) 
        if 'viral_load_cat' in self.merged.columns:
            print('controlling for viral load')
            self.viral_load_analysis()
        else:
            print(
        '''There appears to be no viral load feature, 
            therefore cannot investigate viral load effect on metrics''')
        self.save_metrics()
        self.summarise() 
    def genMetrics(self, data):
        '''
        run all the desired evaluation metrics
        data: pd.Dataframe
            contains the subset of data you want to generated metrics from 
        '''
        print('Running metric analysis on:')
        print(self.type_analysis, data.shape)
        self.metrics[self.type_analysis] = {}
        self.metrics[self.type_analysis]['confusion matrix'] = self.confusion_matrix(data).tolist()
        self.metrics[self.type_analysis]['recall'] = self.recall(data)
        self.metrics[self.type_analysis]['precision'] = self.precision(data)
        self.metrics[self.type_analysis]['classification report'] = self.classification_report(data)
        self.roc_auc(data)
        self.pr_auc(data)

    def check_location_path(self):
        if not os.path.exists(self.location):
            raise f'{self.location} does not exist. Create the directory and re run'
    
    def check_labels_agree(self):
        '''
        do the meta data labels and the labels from the modelling script
        just a sanity check 
        '''
        assert any(self.merged['labels'] == self.merged['test_result']), \
                f'Disagreement between the labels passed to the model and meta data'
    
    def check_labels_consistent(self):
        '''
        The order of samples is shuffled to match the meta data on merge. It
        is therefore important to check that the labels agree/ the shuffle maintained
        sample way matching
        '''
        pd.testing.assert_frame_equal(self.predictions[['barcode',
                                                        'labels',
                                                        'preds',
                                                        'logits']],
                                      self.merged[['barcode',
                                                  'labels',
                                                  'preds',
                                                  'logits']]
                                    )
    def load_geo_data(self):
     
        url = 'https://opendata.arcgis.com/api/v3/datasets/83f458a118604169b599000411f364bf_0/downloads/data?format=shp&spatialRefId=27700'
        self.geo_data = geopandas.read_file(url, ignore_geometry=True)

    def merge_meta_predictions(self):
        '''
        Merge the meta data and the test data keeping only cases in split
        ''' 
        return  self.predictions.merge(self.meta_data,
                                            how='inner',
                                            on='barcode')
        
    def recall(self, data):
        '''
        Unweighted average recall
        '''
        return recall_score(data['test_result'].tolist(),
                            data['preds'].tolist(),
                            average='macro')
    def precision(self, data):
        '''
        Unweighted average precision
        '''
        return precision_score(data['test_result'].tolist(),
                                data['preds'].tolist(),
                                average='macro')
    def confusion_matrix(self, data):
        '''
        create basic confusion matrix for logging to 'metrics' and
        plots a pretty confusion matrix plot and saves it to results folder
        '''
        cm = confusion_matrix(data['test_result'].tolist(),
                                data['preds'].tolist())
        fig, ax = plt.subplots(1,1)
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="YlGnBu")
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
        ax.set_title('Confusion Matrix') 
        ax.xaxis.set_ticklabels(['Negative', 'Positive'])
        ax.yaxis.set_ticklabels(['Negative', 'positive'])
        plt.savefig(f'{self.location}/confusion_matrix_{self.type_analysis}_{self.split}.png',
                    bbox_inches='tight'
                    )
        return cm


    def classification_report(self, data):
        '''
        produce the sklearn classification report
        '''
        return classification_report(data['test_result'].tolist(),
                                    data['preds'].tolist())
    
    def roc_auc(self, data):
        '''
        produce roc curve and savefig
        '''
        if self.classifier == None:
            fpr, tpr, _ = roc_curve(data['test_result'].tolist(),
                                    data['logits'].tolist())
            roc_auc = auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=lw,
                label="ROC curve (area = %0.2f)" % roc_auc,
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc="lower right") 
            plt.title(self.description)
            plt.savefig(f'{self.location}/{self.split}_{self.type_analysis}_roccurve.png',
                        bbox_inches='tight') 
            plt.close()

    def pr_auc(self, data):
        '''
        produce pr curve and savefig
        '''
        if self.classifier == None:
            pr, rec, _ = precision_recall_curve(data['test_result'].tolist(),
                                                data['logits'].tolist())
            pr_auc = auc(rec, pr)
            plt.figure()
            lw = 2
            plt.plot(
                rec,
                pr,
                color="darkorange",
                lw=lw,
                label="PR curve (area = %0.2f)" % pr_auc,
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="lower right") 
            plt.title(self.description)
            plt.savefig(f'{self.location}/{self.split}_{self.type_analysis}_prcurve.png',
                        bbox_inches='tight') 

            plt.close()


    def control_symptom_analysis(self, symptoms):
        '''
        what are the performances when you control for an attribute e.g. symptoms?
        this function will perform evaluation metrics for a subset of the data
        where the symptoms in symptoms are controlled
        args: 
        symptoms --> list of strs which are symptoms in the study to control for.
        '''
        #select subset of the test set which meet your control requirements
        selection = self.merged.symptoms.apply(
                            lambda x: any(symptom in symptoms for symptom in x))
        subset = self.merged[selection]
        self.type_analysis = '_'.join(symptoms)
        #now rerun the genMetrics with the controlled dataset
        self.genMetrics(subset)

    def control_geo_analysis(self, geo):
        '''
        ditto for control_symptom_analysis but for locations
        '''
        
        self.type_analysis = '_'.join(geo)
        geo = [self.map_county_to_lac(i) for i in geo] 
        #select subset of the test set which meet your control requirements
        selection = self.merged.local_authority_code.apply(lambda x:  x in geo)
        subset = self.merged[selection]
        #now rerun the genMetrics with the controlled dataset
        self.genMetrics(subset)

    def viral_load_analysis(self):
        '''
        calculate the performance metrics for 3 levels of viral load
        '''
        subset = self.merged[self.merged['test_result'] == 'Positive']
        subset = subset.dropna(subset=['submission_delay', 'viral_load_cat'])
        subset = subset[subset['submission_delay'].astype('timedelta64[h]') <= 50]
        for viral_cat in subset.viral_load_cat.unique():
            temp_subset = subset[subset['viral_load_cat'] == viral_cat]
            self.type_analysis = viral_cat
            self.genMetrics(temp_subset)

    def map_county_to_lac(self, location):
        '''
        Add a column to the merge df with the corresponding name of the county
        '''
        if location not in self.geo_data['LAD19NM'].unique():
            raise Exception(f"{location} is not a valid code." \
                         f"Please choose from {self.geo_data['LAD19NM'].unique()}")
        return self.geo_data.loc[self.geo_data['LAD19NM'] == location, 'LAD19CD'].iloc[0]

    def save_metrics(self):
        '''
        save the calculated metrics
        '''
        with open(f'{self.location}/metrics.json', 'w') as f:
            json.dump(self.metrics, f)

    def summarise(self):
        '''
        attempts to summaries all the analysis in a series of compartive plots
        '''
        summary = pd.DataFrame.from_dict(self.metrics, orient='index')
        summary.reset_index(inplace=True)
        summary = summary.rename(columns = {'index':'subset'})
        x = np.arange(len(summary))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, summary['recall'].tolist(), width, label='recall')
        rects2 = ax.bar(x + width/2, summary['precision'], width, label='precision')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Score')
        ax.set_title('Model performance after controlling for confounding factors')
        ax.set_xticks(x)
        ax.set_xticklabels(summary['subset'].tolist())
        ax.tick_params(axis='x', rotation=80)
        ax.legend()
        plt.savefig(f'{self.location}/summary_{self.split}.png',
                    bbox_inches='tight'
                    )
    @staticmethod        
    def conf_int(score, n_positive, n_negative, test_type, just_conf=True):
        '''
        calculate confidence intervals for a given metric
        score: Float - the metric score
        n_positive, n_negative: Int - number of test set instances.
        test_type: str - what type of metric to generate the ci for

        method for AUC taken from:https://pubs.rsna.org/doi/epdf/10.1148/radiology.143.1.7063747
        '''
        if test_type in ['ROC-AUC', 'PR-AUC']:
            q_1 = score / ( 2 - score ) 
            q_2 = (2 * score ** 2) / (1 + score)
            numerator = score*(1 - score) + (n_positive - 1)*(q_1 - score**2) + (n_negative-1)*(q_2-score**2)
            se = (numerator/(n_positive * n_negative))**0.5
            if just_conf:
                return 1.960*se

            return score + 1.960*se, score - 1.960*se, 1.906*se
        else:
            conf = 1.960 * ((score*(1 - score))/(n_positive + n_negative))**0.5
            if just_conf:
                return conf
            return score + conf, score - conf, conf

class AUCSVM():
    def __init__(self, estimator, X, y):
        #Generate confidence scores for samples - this is proportional to distance of the samples
        # from the hyperplane
        self.estimator = estimator
        self.X = X
        self.y = y
        self.y_preds = estimator.decision_function(X)
    
    def PR_AUC(self, **kwargs):
        prec, rec, _ = precision_recall_curve(self.y, self.y_preds, pos_label='Positive')
        pr_auc = average_precision_score(self.y, self.y_preds, pos_label='Positive')
        #now plot
        viz = PrecisionRecallDisplay(recall=rec,
                precision=prec,
                average_precision=pr_auc)
        return viz.plot(**kwargs), prec, rec, pr_auc
    
    def ROC_AUC(self, **kwargs):
        fpr, tpr, _ = roc_curve(self.y, self.y_preds, pos_label='Positive')
        roc_auc = auc(fpr, tpr)
        #now plot
        viz = RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc)
        return viz.plot(**kwargs), fpr, tpr, roc_auc 

if __name__ == '__main__':
    sys.path.append('../')
    scores = [
            ('naive', 0.763, 0.846, 0.774, 3514, 6663),
            ('Standard test', 0.733, 0.8, 0.684, 3820, 7301),
            ('Matched test', 0.594, 0.619, 0.594, 907, 907),
            ('matched train Matched test', 0.602, 0.635, 0.626, 907, 907),
            ('Longitudinal', 0.739, 0.818, 0.715, 10315, 20509),
            ('Matched Longitudinal', 0.583, 0.621, 0.594, 2098, 2098), 
            ('matched train Matched Longitudinal', 0.572, 0.604, 0.579, 2098, 2098)] 

    for score in scores:
        cis = EvalMetrics.conf_int(score[2], score[4], score[5], 'ROC-AUC', just_conf=False)
        print(f'{score[0]}: ROC-AUC: {score[2]} +ci: {cis[0]} -ci: {cis[1]} plusminus{cis[2]}')

        cis = EvalMetrics.conf_int(score[3], score[4], score[5], 'PR-AUC', just_conf=False)
        print(f'{score[0]}: PR-AUC: {score[3]} +ci: {cis[0]} -ci: {cis[1]} plusminus{cis[2]}')
        
        cis = EvalMetrics.conf_int(score[1], score[4], score[5], 'UAR', just_conf=False)
        print(f'{score[0]}: UAR: {score[1]} +ci: {cis[0]} -ci: {cis[1]} plusminus{cis[2]}')
