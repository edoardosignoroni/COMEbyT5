# NOTES

## RUNNING EXPERIMENTS

+ xlm-r batch size 16     1

+ canine-c-lamb-0.018     2   batch_size 1024
+ canine-c-lamb-0.001     3   batch_size 1024
+ canine-c-lamb-0.0001    5   batch_size 1024

+ canine-c-0.3            7   possible OOM
+ canine-c-1.0            8
+ canine-c-big_estimator  9

+ 8/32 = 256 batch size
+ last layer
+ no layerwise lr decay
+ 1536, 768 estimator
+ default learning rate
+ unfrozen

## RELATED WORK

+ Canine
  + token free (chars to unicode codepoints)
  + character level model
  + same train data of mBERT (104 langs wikipedia) on MLM and NSP
  + a char embedding layer > downsampling > same core of 12 transformer layers > upsampling
  + pretrained with char loss on random char masks or subword loss on random subword masks
  + results in token free model in any case

+ COMET (WMT 2020 submission)
  + 

+ MetricX-23
  + Learned regression metric based on mT5-XXL encoder-decoder
  + Finetuned on DA and then MQM
  + z-norm DA scores and raw scores is a trade off between segment and system level performance
  + raw MQM training is better than z-norm ones
  + Metric performance increases with the size of the model
  + input = candidate: hypothesis reference: reference. concatenated (NO SOURCE?)
  + arbitrary "<extra_id_**>" token for the decoder start of sentence
  + train DA 2015-2020 valid 2022
  + train MQM 2020-2021 valid 2022 (negate the scores to [-25;0]) higher scores = no errors
  + 1/2% of synthetic data (ref=ref=100 and ''=ref=0) is a good tradeoff for DEMETR and keeping overall performance
  + DA finetune is better for system level; MQM fine-tune is better for segment-level


## TODO

