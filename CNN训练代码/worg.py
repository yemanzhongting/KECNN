# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import hashlib
import math
import os
import random
import sys
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

data_index = 0

def word2vec_basic(log_dir):
  """Example of building, training and visualizing a word2vec model."""
  # Create the directory for TensorBoard variables if there is not.
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  # # Step 1: Download the data.
  # # Note: Source website does not support HTTPS right now.
  # url = 'http://mattmahoney.net/dc/'
  #
  # # pylint: disable=redefined-outer-name
  # def maybe_download(filename, expected_bytes):
  #   """Download a file if not present, and make sure it's the right size."""
  #   if not os.path.exists(filename):
  #     filename, _ = urllib.request.urlretrieve(url + filename,filename)
  #   #获取文件的相关属性信息
  #   statinfo = os.stat(filename)
  #   #判断文件大小是否相等
  #   if statinfo.st_size == expected_bytes:
  #     print('Found and verified', filename)
  #   else:
  #     print(statinfo.st_size)
  #     raise Exception('Failed to verify ' + filename +'. Can you get to it with a browser?')
  #   return filename

  #filename = maybe_download('text8.zip',31344016)
  filename=r'D:\程序\Text-classification-CNN\text8.zip'
  # Read the data into a list of strings.
  def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

  vocabulary = read_data(filename)    #list()列表
  print('Data size', len(vocabulary))

  # Step 2: Build the dictionary and replace rare words with UNK token.
  vocabulary_size = 50000

  def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]] #二维数组
    count.extend(collections.Counter(words).most_common(n_words - 1)) #截取前49999个高频词
    #dictionary为{}，key为word value为index
    dictionary = {word: index for index, (word, _) in enumerate(count)} #不需要但又必须定义的变量点以为“_”
    data = []
    unk_count = 0
    for word in words:
      index = dictionary.get(word, 0)
      if index == 0:  # dictionary['UNK']  最后统计下所有的低频词UNK的个数
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #反转字典 key为index value为word
    return data, count, dictionary, reversed_dictionary

  # Filling 4 global variables:（获取训练集中的信息，保存在下面的全局变量中）
  # data - list of codes (integers from 0 to vocabulary_size-1).
  #   This is the original text but words are replaced by their codes
  # count - map of words(strings) to count of occurrences
  # dictionary - map of words(strings) to their codes(integers)
  # reverse_dictionary - map of codes(integers) to words(strings)
  data, count, unused_dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
  del vocabulary  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  # Step 3: Function to generate a training batch for the skip-gram model.
  def generate_batch(batch_size, num_skips, skip_window):
    global data_index   #代表目前训练数据段的其实位置
    assert batch_size % num_skips == 0  #断言 如果batch_size % num_skips!=0 程序报错中断
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)  #batch内为8个word的数字索引
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin（双向队列）
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span]) #data[0:3] 训练数据中索引为1，2，3的word，数据只在这有，其余的都为索引
    data_index += span
    for i in range(batch_size // num_skips):  #整除 4 i:0,1,2,3
      context_words = [w for w in range(span) if w != skip_window]  #获取上下文的word [0,2]
      words_to_use = random.sample(context_words, num_skips)  #[0,2]  ？
      for j, context_word in enumerate(words_to_use): #j:0,1  context:0,2
        batch[i * num_skips + j] = buffer[skip_window]  #batch[n] n:"0,1","2,3","4,5","6,7"每两个batch（target_word）内数据相同
        labels[i * num_skips + j, 0] = buffer[context_word] #labels[n,0] n:0,1,2,3,4,5,6,7  设计的还是很巧妙的
        #labels稍微复杂一些，labels[0,0]=buffer[0] labels[1,0]=buffer[2] labels[2,0]=buffer[1] lables[3,0]=buffer[3]...
      if data_index == len(data):
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index]) #这里会改buffer内的值
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index - span) % len(data)
    return batch, labels

  #batch_size训练一批单词的个数、num_skips
  batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
  for i in range(8):  #测试下效果
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
          reverse_dictionary[labels[i, 0]])

  # Step 4: Build and train a skip-gram model.

  batch_size = 128
  embedding_size = 128  # Dimension of the embedding vector.
  skip_window = 1  # How many words to consider left and right.
  num_skips = 2  # How many times to reuse an input to generate a label.
  num_sampled = 64  # Number of negative examples to sample. 负采样：减小计算量，达到较好效果的一种方式

  # We pick a random validation set to sample nearest neighbors. Here we limit
  # the validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent. These 3 variables are used only for
  # displaying model accuracy, they don't affect calculation.
  valid_size = 16  # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)   #验证集

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/gpu:0'):#'/cpu:0'
      # Look up embeddings for inputs.
      with tf.name_scope('embeddings'):
        embeddings = tf.Variable(     #定义 词向量 维度为：50000*128  50000个词，每个词128个维度
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        #抽取要训练的词，train_inputs就是要训练的词，训练哪些就从embeddings中抽取出来
        #embedding_lookup(params, ids),比如说ids=[1,7,4],就是返回params中的1，7，4行组成的tensor
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss  噪声对比工具（负采样）
      with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
      with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss and why choosing NCE over tf.nn.sampled_softmax_loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    #   http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(    #二次代价函数
          tf.nn.nce_loss(       #nce负采样
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)   #梯度下降法

    # Compute the cosine similarity between minibatch examples and all 余弦相似度比较测试集数据
    # embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

  # Step 5: Begin training.
  num_steps = 100001

  with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(batch_size, num_skips,skip_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # Define metadata variable.
      run_metadata = tf.RunMetadata()

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      # Also, evaluate the merged op to get all summaries from the returned
      # "summary" variable. Feed metadata variable to session for visualizing
      # the graph in TensorBoard.
      _, summary, loss_val = session.run([optimizer, merged, loss],
                                         feed_dict=feed_dict,
                                         run_metadata=run_metadata)
      average_loss += loss_val

      # Add returned summaries to writer in each step.
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word     #打印16个测试集中前8个比较相似的词，好的词向量模型 比较相似的词的余弦距离也是比较相近的

          print(log_str,', '.join([reverse_dictionary[nearest[k]] for k in range(top_k)]))
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in xrange(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

  writer.close()

  # Step 6: Visualize the embeddings.

  # pylint: disable=missing-docstring
  # Function to draw visualization of distance between embeddings.
  def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')

    plt.savefig(filename)

  try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE   #把词向量通过TSNE降维的方式给画出来
    import matplotlib
    #matplotlib.use("Agg")
    matplotlib.use("Pdf")
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

  except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)


# All functionality is run after tf.compat.v1.app.run() (b/122547914). This
# could be split up but the methods are laid sequentially with their usage for
# clarity.
def main(unused_argv):
  # Give a folder path as an argument with '--log_dir' to save
  # TensorBoard summaries. Default is a log folder in current directory.
  current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')
  flags, unused_flags = parser.parse_known_args()
  word2vec_basic(flags.log_dir)


if __name__ == '__main__':
  tf.app.run()