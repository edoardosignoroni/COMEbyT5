from datasets import load_dataset

data_dir = "/nlp/projekty/mtlowre/COMEbyT5/data"

#pairs = ["en-cs", "en-de", "en-fi", "en-gu", "en-kk", "en-lt", "en-ru", "en-zh"]
pairs = ["de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"]
dataset = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train")
dataset = dataset.remove_columns(["raw","annotators"])

for lp in pairs:
    pair_dataset = dataset.filter(lambda example: example["lp"] in lp)
    pair_dataset = pair_dataset.filter(lambda example: example["domain"] == "news")
    pair_dataset = pair_dataset.filter(lambda example: example["year"] == 2019)
    pair_dataset = pair_dataset.remove_columns(['year', 'domain'])

    pair_dataset.to_csv(f'{data_dir}/wmt-2019-{lp}.csv')
