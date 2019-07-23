from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField,SequenceLabelField
'''
在AllenNLP中，我们将每个训练实例（example）表示为包含各种类型的字段的Instance。
这里每个实例(example)都有一个包含句子的TextField，
以及一个包含相应词性标签的SequenceLabelField
'''

from allennlp.data.dataset_readers import DatasetReader
'''
通常使用AllenNLP来解决这样的问题，您必须实现两个类(class)。
第一个是DatasetReader，它包含用于读取数据文件和生成实例流(Instances)的逻辑。
'''

from allennlp.common.file_utils import cached_path
'''
我们经常要从URL加载数据集或模型。 cached_pa​​th帮助程序下载此类文件，在本地缓存它们，并返回本地路径。它还接受本地文件路径（它只是按原样返回）。
'''

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
'''
有多种方法可以将单词表示为一个或多个索引。例如，您可以维护唯一单词的词汇表，并为每个单词指定相应的ID。
或者你可能在单词中每个字符有一个id，并将每个单词表示为一系列id。 AllenNLP使用具有TokenIndexer抽象的表示。
'''

from allennlp.data.vocabulary import Vocabulary
'''
TokenIndexer表示如何将token转换为索引的规则，而Vocabulary包含从字符串到整数的相应映射。
例如，您的token索引器可能指定将token表示为字符ID序列，在这种情况下，Vocabulary将包含映射{character - > id}。
在另外一个特定的例子中，我们使用SingleIdTokenIndexer为每个token分配一个唯一的id，因此Vocabulary只包含一个映射{token - > id}（以及反向映射）。
'''

from allennlp.models import Model
'''
除了DatasetReader之外，你通常需要实现的另一个类是Model，
它是一个PyTorch模块，它接受张量输入并产生张量输出的dict（包括你想要优化的训练损失）。
'''

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
'''
如上所述，我们的模型将包含一个嵌入层(embedding layer)，然后是LSTM，然后是前馈层。 
AllenNLP包括所有这些智能处理填充(padding)和批处理(batching)的抽象，以及各种实用功能。
'''

from allennlp.training.metrics import CategoricalAccuracy
'''
我们希望跟踪训练（training）和验证（validation）数据集的准确性（accuracy）。
'''

from allennlp.data.iterators import BucketIterator
'''
在我们的训练中，我们需要一个可以智能地批量处理数据的DataIterators。
'''

from allennlp.training.trainer import Trainer
'''
我们将使用AllenNLP的全功能training。
'''

from allennlp.predictors import SentenceTaggerPredictor
'''
最后，我们想要对新输入做出预测，下面将详细介绍。
'''
torch.manual_seed(1)
'''
设定生成随机数的种子，并返回一个torch._C.Generator对象
'''


# ===================一、我们需要编写的准备数据集类======================
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers:Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        '''
        我们的DatasetReader需要的唯一参数是TokenIndexers的dict，它指定如何将tokens转换为索引。
        默认情况下，我们只为每个token（我们称之为“tokens”）生成一个索引，这只是每个不同token的唯一ID。 
        （这只是您在大多数NLP任务中使用的标准“索引词”映射。）
        '''


    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)
    '''
    DatasetReader.text_to_instance获取与训练example相对应的输入（在这种情况下是句子的tokens和相应的词性标签(part-of-speech tags)），
    实例化相应的Fields（在这种情况下是句子的TextField和其标签的SequenceLabelField），并返回包含这些字段(Fields)的实例(Instance)。
    请注意，tags是可选的，因为我们希望能够从未标记的数据创建实例(instances)以对它们进行预测。 
    '''

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)

        '''
        我们必须实现的另一个部分是_read，它接受一个文件名并生成一个实例流(Instances)。大部分工作已经在text_to_instance中完成。
        '''


# ===================二、我们需要编写的模型部分（Model）类======================
class LstmTagger(Model):
    '''
    您基本上必须实现的另一个类是Model，它是torch.nn.Module的子类。
    它的工作原理在很大程度上取决于你，它主要只是需要一个前向方法(forward method)，它接受张量输入并产生一个张量输出的字典，
    其中包括你用来训练模型的损失(losss)。
    如上所述，我们的模型将包括嵌入层(embedding layer)，序列编码器(sequence encoder)和前馈网络(feedforward network)。
    '''

    '''
    可能看似不寻常的一件事是我们将嵌入器（embedder）和序列编码器(sequence encoder)作为构造函数参数(constructor parameters)传递。
    这使我们可以尝试不同的嵌入器(embedders)和编码器(encoders)，而无需更改模型代码。
    '''
    def __init__(self,

                 word_embeddings: TextFieldEmbedder,
                 # 嵌入层(embedding layer)被指定为AllenNLP TextFieldEmbedder，它表示将tokens转换为张量(tensors)的一般方法。
                 # （这里我们知道我们想要用学习的张量来表示每个唯一的单词，但是使用通用类(general class)可以让我们轻松地尝试不同类型的嵌入，例如ELMo。）

                 encoder: Seq2SeqEncoder,
                 # 类似地，编码器(encoder)被指定为通用Seq2SeqEncoder，即使我们知道我们想要使用LSTM。
                 # 同样，这使得可以很容易地尝试其他序列编码器(sequence encoders)，例如Transformer。

                 vocab: Vocabulary) -> None:
                 # 每个AllenNLP模型还需要一个词汇表(Vocabulary)，其中包含tokens到索引(indices)和索引标签(labels to indices)的命名空间映射。

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        # 请注意，我们必须将vocab传递给基类构造函数(base class constructor)。

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        # 前馈层(feed forward layer)不作为参数传入，而是由我们构造。
        # 请注意，它会查看编码器(encoder)以查找正确的输入维度并查看词汇表(vocabulary)（特别是在 label->index 映射处）以查找正确的输出维度。

        self.accuracy = CategoricalAccuracy()
        # 最后要注意的是我们还实例化了一个CategoricalAccuracy指标，我们将用它来跟踪每个训练(training)和验证(validation)epoch的准确性。

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    # 接下来我们需要实现forward，这是实际计算发生的地方。数据集中的每个实例(Instance)都将（与其他实例(instances)一起批处理）输入forward。
    # 张量的输入作为forward方法的输入，并且它们的名称应该是实例（Instances）中字段（fields）的名称。
    # 在这种情况下，我们有一个句子字段(sentence field)和（可能）标签字段(labels field)，所以我们将相应地构建我们的forward：

        mask = get_text_field_mask(sentence)
        # AllenNLP设计用于批量输入，但不同的输入序列具有不同的长度。
        # 因此，AllenNLP填充（padding）较短的输入，以便批处理具有统一的形状，这意味着我们的计算需要使用掩码(mask)来排除填充。
        # 这里我们只使用效用函数(utility function) get_text_field_mask，它返回与填充和未填充位置相对应的0和1的张量。

        embeddings = self.word_embeddings(sentence)
        # 我们首先将句子张量（每个句子一系列tokens ID）传递给word_embeddings模块，该模块将每个句子转换为嵌入式张量序列(a sequence of embedded tensors)。

        encoder_out = self.encoder(embeddings, mask)
        # 接下来，我们将嵌入式张量(embedded tensors)（和掩码(mask)）传递给LSTM，LSTM产生一系列编码(encoded)输出。

        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        # 最后，我们将每个编码输出张量(encoded output tensor)传递给前馈层(feedforward)，以产生对应于各种标签(tags)的logits。

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        # 和以前一样，标签是可选的，因为我们可能希望运行此模型来对未标记的数据进行预测。
        # 如果我们有标签，那么我们使用它们来更新我们的准确度指标（accuracy metric）并计算输出中的“损失(loss)”。

        return output

        def get_metrics(self, reset: bool = False) -> Dict[str, float]:
            return {"accuracy": self.accuracy.get_metric(reset)}
        # 我们提供了一个准确度指标(accuracy metric)，每个正向传递都会更新。
        # 这意味着我们需要覆盖(override)从中提取数据的get_metrics方法。
        # 这意味着，CategoricalAccuracy指标存储预测数量和正确预测的数量，在每次call to forward 期间更新这些计数。
        # 每次调用get_metric都会返回计算的精度，并（可选）重置计数，这使我们能够重新跟踪每个epoch的准确性。


# =====================  正式开始  ==========================
reader = PosDatasetReader()
# 现在我们已经实现了数据集读取器(DatasetReader)和模型(Model)，我们已准备好进行训练。我们首先需要一个数据集读取器的实例(instance)。

train_dataset = reader.read("data/training.txt")
validation_dataset = reader.read("data/validation.txt")
# 我们可以使用它来读取训练数据和验证数据。这里我们从URL读取它们，但如果您的数据是本地的，您可以从本地文件中读取它们。
# 我们使用cached_pa​​th在本地缓存文件（以及处理reader.read到本地缓存版本的路径。）

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# 一旦我们读入了数据集，我们就会使用它们来创建我们的词汇表（vocabulary）（即从 tokens/labels 到 ids 的 映射[s]）。

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
# 现在我们需要构建模型。我们将为嵌入层(embedding layer)和 LSTM 的隐藏层(hidden layer)选择一个大小(size)。

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
# 对于embedding the tokens，我们将使用BasicTextFieldEmbedder，它从索引名称到嵌入(embeddings)进行映射。
# 如果你回到我们定义DatasetReader的地方，默认参数包括一个名为“tokens”的索引，所以我们的映射只需要一个对应于该索引的嵌入(embedding)。
# 我们使用词汇表(vocabulary)来查找我们需要多少嵌入(embeddings)，并使用EMBEDDING_DIM参数来指定输出维度。
# 也可以从预先训练的嵌入开始（例如，GloVe向量），但是没有必要在这个小玩具数据集上做到这一点。

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
# 接下来我们需要指定序列编码器(sequence encoder)。
# 这里对PytorchSeq2SeqWrapper的需求有点不幸（如果你使用配置文件就不用担心了），
# 但是这里需要为内置的PyTorch模块添加一些额外的功能（和更简洁的接口）。
# 在AllenNLP中，我们首先完成所有批处理，因此我们也指定了它。

model = LstmTagger(word_embeddings, lstm, vocab)
# 最后，我们可以实例化 Model。

if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
# 如果由DPU就使用GPU0，没有就不使用

optimizer = optim.SGD(model.parameters(), lr=0.1)
# 现在我们已经准备好训练模型了。我们需要的第一件事是优化器（optimizer）。我们可以使用PyTorch的随机梯度下降（ stochastic gradient descent）。

iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
# 我们需要一个DataIterator来处理我们数据集的批处理。 BucketIterator按指定字段对实例（instances）进行排序，以创建具有相似序列长度的批次(batches)。
# 这里我们指出我们想要通过句子字段(sentence field)中的tokens数对实例(instances)进行排序。

iterator.index_with(vocab)
# 我们还指定迭代器(iterator)应确保使用我们的词汇表(vocab)对其实例(instances)进行索引;
# 也就是说，他们的字符串已经使用我们之前创建的映射转换为整数。

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)
# 现在我们实例化我们的Trainer并运行它。
# 在这里，我们告诉它运行1000个epochs并且如果它花费10个epochs而没有验证指标（validation metric）提升则提前停止训练。
# 默认验证指标（validation metric）标准是损失（通过变小来改善），但也可以指定不同的指标和方向（例如，精度（accuracy）应该变大）。

trainer.train()
# 当我们启动它时，它将为每个epochs打印一个包含“损失(loss)”和“准确度(accuracy)”度量标准的进度条。
# 如果我们的模型是好的，那么损失应该降低，并且在我们训练时准确度会提高。

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
# 与最初的PyTorch教程一样，我们希望查看模型生成的预测。
# AllenNLP包含一个Predictor抽象，它接受输入，将它们转换为实例(instances)，输入模型，并返回JSON可序列化的结果。
# 通常你需要实现自己的Predictor，但AllenNLP已经有一个SentenceTaggerPredictor在这里完美运行，所以我们可以使用它。
# 它需要我们的模型（用于进行预测）和数据集读取器（用于创建实例）。

# ============  预测 ==========================
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
# 它有一个只需要一个句子的预测方法，并从前向（forward）返回输出字典（JSON可序列化版本）。
# 这里tag_logits将是（5X3）的logits数组，对应于5个单词中每个单词的3个可能标签。

tag_ids = np.argmax(tag_logits, axis=-1)
# 为了获得实际的“预测”，我们可以采用argmax。

print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
# 然后使用我们的词汇表来查找预测的标签。


# ===========  保存模型 ========================
# 最后，我们希望能够保存我们的模型并在以后重新加载它。我们需要保存两件事。
with open("./tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
# 首先是模型权重

vocab.save_to_files("./tmp/vocabulary")
# 然后是vacabulary

# ===========  从新加载模型 ====================
vocab2 = Vocabulary.from_files("./tmp/vocabulary")
# 我们只保存了模型权重，因此如果我们想重用它们，我们实际上必须使用代码重新创建相同的模型结构。
# 首先，让我们将词汇表重新加载到一个新变量中。

model2 = LstmTagger(word_embeddings, lstm, vocab2)
# 然后让我们重新创建模型（如果我们在不同的文件中执行此操作，我们当然必须重新实例化嵌入字(word_embeddings)和lstm）。

with open("./tmp/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))
# 之后我们必须加载它的状态。

if cuda_device > -1:
    model2.cuda(cuda_device)
# 在这里，我们将加载的模型移动到我们之前使用的GPU。
# 这是必要的，因为我们之前使用原始模型移动了word_embeddings和lstm。
# 所有模型的参数都需要在同一设备上。

predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
# 预测部分是一样的





































