import tensorflow as tf
import argparse
from data_utils import load_data, create_mapping, get_batches, Batch, sentences_to_ids
from seq2seq_tensorflow import Seq2SeqModel
import os
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
# configurations for file
parser.add_argument("--source_file", default="./data/small_vocab_en", type=str, help="Path for source data")
parser.add_argument("--target_file", default='./data/small_vocab_fr', type=str, help="Path for target data")
parser.add_argument("--model_dir", default='model/',  type=str,  help="Path to save model checkpoints")
# configurations for model
parser.add_argument("--rnn_size", default=50,  type=int,
                    help="Number of hidden units in each layer")
parser.add_argument("--num_layers", default=2,  type=int,
                    help="Number of layers in each encoder and decoder")
parser.add_argument("--embedding_size", default=50,  type=int,
                    help="Embedding dimensions of encoder and decoder inputs")
parser.add_argument("--learning_rate", default=0.001,  type=float, help="Learning rate")
parser.add_argument("--lr_decay",   default=0.9,  type=float, help="Initial learning rate decay")
parser.add_argument("--numEpochs", default=100,  type=int, help="Maximum # of training epochs")
parser.add_argument("--batch_size", default=128,  type=int,  help="Batch size")
parser.add_argument("--steps_per_checkpoint", default=10,  type=int,
                    help="Save model checkpoint every this iteration")
parser.add_argument("--summary_dir", default='summary/',  type=str,  help="mode name")

args = parser.parse_args()

source_data = load_data(args.source_file)
target_data = load_data(args.target_file)

source_idx_to_word, source_word_to_idx = create_mapping(source_data)
target_idx_to_word, target_word_to_idx = create_mapping(target_data)


source_data_idx, target_data_idx = sentences_to_ids(source_data, target_data,source_word_to_idx,
                                                    target_word_to_idx)
source_data_train, source_data_dev, target_data_train, target_data_dev\
    = train_test_split(source_data_idx, target_data_idx, test_size=0.3, random_state=1,
                       shuffle=False, stratify=None)

with tf.Session() as sess:
    model = Seq2SeqModel(rnn_size=args.rnn_size,
                         num_layers=args.num_layers,
                         enc_embedding_size=args.embedding_size,
                         learning_rate=args.learning_rate,
                         dec_embedding_size=args.embedding_size,
                         source_word_to_idx=source_word_to_idx,
                         target_word_to_idx=target_word_to_idx,
                         mode='train',
                         use_attention=False,
                         beam_search=False,
                         beam_size=5,
                         max_gradient_norm=5.0
                         )

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())

    train_current_step = 0
    dev_current_step = 0
    train_writer = tf.summary.FileWriter(args.summary_dir+'train', graph=sess.graph)
    dev_writer = tf.summary.FileWriter(args.summary_dir+'dev')
    train_batch = Batch()
    dev_batch = Batch()
    for e in range(args.numEpochs):
        dev_loss_sum = 0
        print("----- Epoch {}/{} -----".format(e + 1, args.numEpochs))
        for batch_i, (train_source_batch, train_target_batch, train_source_length, train_target_length) in enumerate(
                get_batches(source_data_train, target_data_train, args.batch_size, source_word_to_idx['<PAD>'])):

            train_batch.encoder_inputs = train_source_batch
            train_batch.decoder_targets = train_target_batch
            train_batch.encoder_inputs_length = train_source_length
            train_batch.decoder_targets_length = train_target_length

        # # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        # for nextBatch in tqdm(batches, desc="Training"):
            train_loss, train_summary = model.train(sess, train_batch)

            train_current_step += 1
            # 每多少步进行一次保存
            if train_current_step % args.steps_per_checkpoint == 0:

                print("----- epoch %d -- step %d --train_loss %.2f" %
                      (e+1, batch_i+1, train_loss))

                train_writer.add_summary(train_summary, train_current_step)

        for batch_i, (dev_source_batch, dev_target_batch, dev_source_length, dev_target_length) in enumerate(
                get_batches(source_data_dev, target_data_dev, args.batch_size,
                            source_word_to_idx['<PAD>'])):

            dev_batch.encoder_inputs = dev_source_batch
            dev_batch.decoder_targets = dev_target_batch
            dev_batch.encoder_inputs_length = dev_source_length
            dev_batch.decoder_targets_length = dev_target_length

            dev_loss, dev_summary = model.eval(sess, dev_batch)
            dev_current_step += 1
            dev_loss_sum += dev_loss
            # 每多少步进行一次保存
            if dev_current_step % args.steps_per_checkpoint == 0:

                dev_writer.add_summary(dev_summary, dev_current_step)
        print("----- epoch %d --mean_dev_loss %.2f" %
              (e + 1, dev_loss_sum/(batch_i+1)))
        model.saver.save(sess, os.path.join(args.model_dir,'en2fr.ckpt'), global_step=train_current_step)
