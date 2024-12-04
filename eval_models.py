#imports
from comet import models as m
from datasets import load_dataset
import torch
import pandas as pd
from scipy.stats import stats

# functions
def filter_dataset(dataset, pair):
    pair_dataset = dataset.filter(lambda example: example["lp"] in pair)
    pair_dataset = pair_dataset.filter(lambda example: example["domain"] == "news")
    pair_dataset = pair_dataset.filter(lambda example: example["year"] == 2019)
    pair_dataset = pair_dataset.remove_columns(['year', 'domain'])
    return pair_dataset

def compute_comet(model, data):
    model_output = model.predict(data, batch_size=16, gpus=1, progress_bar=False)
    return model_output

def compute_correlations(model, data):
    """Computes spearmans correlation coefficient."""
    comet_scores = compute_comet(model, data)
    human_scores = data['score']
 
    preds = comet_scores.scores
    target = human_scores

    kendall = stats.kendalltau(preds, target)
    spearman = stats.spearmanr(preds, target)
    pearson = stats.pearsonr(preds, target)
    report = {
        "kendall": kendall,
        "spearman": spearman,
        "pearson": pearson,
    }
    return report

# load data
print("LOADING DATA FROM HUGGINGFACE...")
dataset = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train")
dataset = dataset.remove_columns(["raw","annotators"])

# load models
print("LOADING MODELS...")
#canine_c = m.load_from_checkpoint('/nlp/projekty/mtlowre/COMEbyT5/_models/canine-c/epoch=4-step=7589-val_kendall=0.284.ckpt')
#canine_s = m.load_from_checkpoint('/nlp/projekty/mtlowre/COMEbyT5/_models/canine-s/epoch=4-step=7589-val_kendall=0.275.ckpt')
#mbert = m.load_from_checkpoint('/nlp/projekty/mtlowre/COMEbyT5/_models/mbert/epoch=3-step=5920-val_kendall=0.329.ckpt')
#xlmr = m.load_from_checkpoint('/nlp/projekty/mtlowre/COMEbyT5/_models/xlmr/epoch=4-step=7589-val_kendall=0.353.ckpt')
canine_c_lamb = m.load_from_checkpoint('/nlp/projekty/mtlowre/COMEbyT5/_models/canine-c-lamb-0.001/epoch=1-step=721-val_kendall=0.299.ckpt')

models = {
    #canine_c : 'canine-c', 
    #canine_s : 'canine-s', 
    #mbert : 'mbert', 
    #xlmr : 'xlmr',
    canine_c_lamb : 'canine-c-lamb'
    }

pairs = ["en-cs", "en-de", "en-fi", "en-gu", "en-kk", "en-lt", "en-ru", "en-zh", "de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"
         ]

# eval for each lang pair

results = pd.DataFrame(columns=['model', 'pair', 'kendall', 'kendall_p', 'spearman', 'spearman_p', 'pearson', 'pearson_p'])

for pair in pairs:
    pair_dataset = filter_dataset(dataset, pair)
    for model in models.keys():
        print(f'EVALUATING {models[model].upper()} ON {pair.upper()}...')
        #print(compute_correlations(model, pair_dataset))
        pair_results = compute_correlations(model, pair_dataset)
        pair_kendall = pair_results['kendall']
        pair_spearman = pair_results['spearman']
        pair_pearson = pair_results['pearson']

        row = pd.DataFrame({
            'model' : [models[model]], 
            'pair' : [pair], 
            'kendall' : [pair_kendall.correlation], 
            'kendall_p' : [pair_kendall.pvalue], 
            'spearman' : [pair_spearman.correlation], 
            'spearman_p' : [pair_spearman.pvalue], 
            'pearson' : [pair_pearson.statistic], 
            'pearson_p' : [pair_pearson.pvalue]
            })
        
        print(row)
        results = pd.concat([results, row], ignore_index=True)

results.to_csv('/nlp/projekty/mtlowre/COMEbyT5/results.tsv', sep='\t', index=False)