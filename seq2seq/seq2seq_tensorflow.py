import tensorflow as tf

class Seq2SeqModel():
    def __init__(self, rnn_size, num_layers, enc_embedding_size,dec_embedding_size,learning_rate, source_word_to_idx,
                 target_word_to_idx, mode, use_attention,
                 beam_search, beam_size, max_gradient_norm=5.0):
        self.learing_rate = learning_rate
        self.enc_embedding_size = enc_embedding_size
        self.dec_embedding_size = dec_embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.source_word_to_idx = source_word_to_idx
        self.target_word_to_idx = target_word_to_idx
        self.source_vocab_size = len(self.source_word_to_idx)
        self.target_vocab_size = len(self.target_word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm

        #add placehoder for model
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None],
                                                     name='decoder_targets_length')
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length,
                                                        name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length,
                                     dtype=tf.float32, name='masks')

        # loss of the model
        self.loss = self.loss_layer()
        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

        # Calculate and clip gradients
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learing_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_params))

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size,initializer=
            tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            #添加dropout
            # 如果仅仅是对RNN输入层与LSTM1层做dropout，就相当于对 encoder_embed_input 做dropout
            # 可以使用 encoder_embed_input=tf.nn.dropout(encoder_embed_input,keep_prob)实现
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, input_keep_prob=self.keep_prob)
            return cell
        #列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def encoder_layer(self):
        with tf.variable_scope('encoder'):
            # 创建LSTMCell，两层+dropout
            encoder_cell = self.create_rnn_cell()

            # Encoder embedding
            # encoder_embed_input = tf.contrib.layers.embed_sequence(encoder_input,
            #                                                        vocab_size=source_vocab_size,
            #                                                        embed_dim=encoding_embedding_size)
            # 构建embedding矩阵,encoder用该词向量矩阵
            enc_embedding = tf.get_variable('embedding', [self.source_vocab_size, self.enc_embedding_size])
            encoder_inputs_embedded = tf.nn.embedding_lookup(enc_embedding, self.encoder_inputs)
            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
            # encoder_outputs用于attention，batch_size*encoder_inputs_length*rnn_size,
            # encoder_state用于decoder的初始化状态，batch_size*rnn_szie
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                               encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)
        return encoder_outputs, encoder_state

    def decoder_layer(self):
        with tf.variable_scope('decode'):

            decoder_cell = self.create_rnn_cell()

            _, encoder_state = self.encoder_layer()

            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                    encoder_state, multiplier=self.beam_size)
            else:
                decoder_initial_state = encoder_state

            dec_embedding = tf.get_variable(name='dec_embedding',
                                            dtype=tf.float32,
                                            shape=[self.target_vocab_size, self.dec_embedding_size])
            output_layer = tf.layers.Dense(self.target_vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(
                                               mean=0.0,
                                               stddev=0.1))
            if self.mode == 'train':

                # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
                # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1],
                                          [1, 1])

                # ending = self.decoder_targets
                decoder_input = tf.concat(
                    [tf.fill([self.batch_size, 1], self.target_word_to_idx['<GO>']), ending], 1)

                decoder_inputs_embedded = tf.nn.embedding_lookup(dec_embedding, decoder_input)

                # 训练阶段，使用TrainingHelper+BasicDecoder的组合，这一般是固定的，当然也可以自己定义Helper类，实现自己的功能
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False,
                                                                    name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                   helper=training_helper,
                                                                   initial_state=decoder_initial_state,
                                                                   output_layer=output_layer)
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
                train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.max_target_sequence_length)
                return train_decoder_outputs

            elif self.mode == 'infer':

                start_tokens = tf.ones([self.batch_size], tf.int32) * self.target_word_to_idx['<GO>']
                end_token = self.target_word_to_idx['<eos>']
                # decoder阶段根据是否使用beam_search决定不同的组合，
                # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码

                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                             embedding=dec_embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embedding,
                                                                               start_tokens=start_tokens,
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)
                infer_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.max_target_sequence_length)

                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果

                if self.beam_search:
                    self.decoder_predict_decode = infer_decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(infer_decoder_outputs.sample_id, -1)

                return infer_decoder_outputs

    def loss_layer(self):
        decoder_outputs = self.decoder_layer()
        # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        # decoder_predict_train 在训练阶段其实没啥用，并不作为t+1阶段的输入。而是直接将target序列
        # 作为decoder的输入
        self.decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1,
                                               name='decoder_pred_train')

        # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志.
        # 这里也可以使用sparse_softmax_cross_entropy_with_logits来计算loss
        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=decoder_outputs, logits=logits)
        # train_loss = (tf.reduce_sum(crossent * target_weights) /
        #               batch_size)
        loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits_train,
                                                     targets=self.decoder_targets,
                                                     weights=self.mask)
        return loss


    def create_feed_dict(self, is_train, is_infer, batch):

        feed_dict = {
            self.encoder_inputs: batch.encoder_inputs,
            self.encoder_inputs_length: batch.encoder_inputs_length,
            self.decoder_targets: batch.decoder_targets,
            self.decoder_targets_length: batch.decoder_targets_length,
            self.keep_prob: 1.0,
            self.batch_size: len(batch.encoder_inputs)
        }

        if is_train:
            feed_dict[self.keep_prob] = 0.5

        if is_infer:
            feed_dict = {
                self.encoder_inputs: batch.encoder_inputs,
                self.encoder_inputs_length: batch.encoder_inputs_length,
                self.keep_prob: 1.0,
                self.batch_size: len(batch.encoder_inputs)
            }

        return feed_dict

    def train(self, sess, batch):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        feed_dict = self.create_feed_dict(is_train=True, is_infer=False, batch=batch)
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op],
                                    feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = self.create_feed_dict(is_train=False, is_infer=False, batch=batch)
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        #infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = self.create_feed_dict(is_train=False, is_infer=True, batch=batch)
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict



    def restore(self, sess, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        print("Reloading the latest trained model...")
        self.saver.restore(sess, dir_model)










# def encoder_layer(encoder_input, source_vocab_size, encoding_embedding_size,
#                   lstm_hidden_size, keep_prob, num_layers, source_sequence_length):
#
#     # Encoder embedding
#     encoder_embed_input = tf.contrib.layers.embed_sequence(encoder_input,
#                                                            vocab_size=source_vocab_size,
#                                                            embed_dim=encoding_embedding_size)
#
#     def get_lstm_cell(hidden_size):
#         lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer=
#         tf.random_uniform_initializer(-0.1, 0.1, seed=2))
#         return lstm_cell
#
#     stacked_cells = tf.contrib.rnn.MultiRNNCell(
#         [tf.contrib.rnn.DropoutWrapper(get_lstm_cell(hidden_size=lstm_hidden_size),
#                                        input_keep_prob=keep_prob)
#          for _ in range(num_layers)])
#
#     encoder_output, encoder_state = tf.nn.dynamic_rnn(stacked_cells, encoder_embed_input,
#                                                       sequence_length=source_sequence_length,
#                                                       dtype=tf.float32)
#     return encoder_output, encoder_state



# def decoder_layer(dec_input, target_vocab_to_int, dec_embedding_size,rnn_size,num_layers,
#                   mode):
#     with tf.variable_scope('decode'):
#         target_vocab_size = len(target_vocab_to_int)
#         dec_embedding = tf.get_variable(name='dec_embedding',
#                                         dtype=tf.float32,
#                                         shape=[target_vocab_size,dec_embedding_size])
#         dec_embed_input = tf.nn.embedding_lookup(dec_embedding, dec_input)
#
#         cells = tf.contrib.rnn.MultiRNNCell(
#             [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
#
#         output_layer = tf.layers.Dense(self.vocab_size,
#                                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,
#                                                                                           stddev=0.1))
#         if mode == 'train':
#             pass
#
#         elif mode == 'infer':
#             pass









