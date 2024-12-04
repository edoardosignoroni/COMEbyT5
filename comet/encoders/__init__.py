# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .bert import BERTEncoder
from .minilm import MiniLMEncoder
from .xlmr import XLMREncoder
from .rembert import RemBERTEncoder
from .xlmr_xl import XLMRXLEncoder
# from .mt5 import MeTric5Encoder
# from .byt5 import ByMeTric5Encoder
# from .mt5_dec import MeTric5EncoderDecoder
from .canine import CanineEncoderForMetric
from .mbert import MBertEncoderForMetric
from .char_bert import CharBERTEncoder

str2encoder = {
    "BERT": BERTEncoder,
    "XLM-RoBERTa": XLMREncoder,
    "MiniLM": MiniLMEncoder,
    "XLM-RoBERTa-XL": XLMRXLEncoder,
    "RemBERT": RemBERTEncoder,
    # "mT5": MeTric5Encoder,
    # "byT5" : ByMeTric5Encoder,
    # "mT5_encdec" : MeTric5EncoderDecoder,
    "Canine" : CanineEncoderForMetric,
    "mbert" : MBertEncoderForMetric,
    "charBERT" : CharBERTEncoder
}