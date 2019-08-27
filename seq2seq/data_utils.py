import codecs
import pickle


class Batch:
    #batch类，里面包含了encoder输入，decoder输入以及他们的长度
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_data(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        data = f.read()

    return data


def create_mapping(data):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    # vocab = set(data.split())
    vocab = list(set([word for line in data.split('\n') for word in line.split()]))
    id_to_word = {idx: word for idx, word in enumerate(special_words+vocab)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    return id_to_word, word_to_id


def get_batches(source_data, target_data, batch_size, pad_tok):

    data_len = len(source_data)
    for i in range(0, data_len, batch_size):
        source_batch = source_data[i:min(i + batch_size, data_len)]
        target_batch = target_data[i:min(i + batch_size, data_len)]
        padded_source_batch, source_batch_length = pad_batch(source_batch, pad_tok)
        padded_target_batch, target_batch_length = pad_batch(target_batch, pad_tok)

        yield padded_source_batch, padded_target_batch, source_batch_length, target_batch_length


def pad_batch(sentence_batch, pad_tok=0):
    sequence_padded, sequence_length = [], []
    max_sentence_length = max([len(sentence) for sentence in sentence_batch])
    for seq in sentence_batch:
        seq = list(seq)
        padding = [pad_tok] * (max_sentence_length - len(seq))
        sequence_padded.append(seq + padding)
        sequence_length.append(len(seq))

    return sequence_padded, sequence_length


def sentences_to_ids(source_text, target_text, source_word_to_id, target_word_to_id):
    # empty list of converted sentences
    source_text_id = []
    target_text_id = []

    # make a list of sentences (extraction)
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")

    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        #这里是因为我发现我的数据中最后面有空行，应该是我自己查看数据时引入的。下面判断就是为了把空行去掉
        if not source_sentences[i]:
            print('the {}th source sentence is none'.format(i))
        elif not target_sentences[i]:
            print('the {}th target sentence is none'.format(i))
        else:
            # extract sentences one by one
            source_sentence = source_sentences[i]
            target_sentence = target_sentences[i]

            # make a list of tokens/words (extraction) from the chosen sentence
            source_tokens = source_sentence.split(" ")
            target_tokens = target_sentence.split(" ")

            # empty list of converted words to index in the chosen sentence
            source_token_id = []
            target_token_id = []

            for index, token in enumerate(source_tokens):
                if (token != ""):
                    source_token_id.append(source_word_to_id.get(token, source_word_to_id.get('<UNK>')))

            for index, token in enumerate(target_tokens):
                if (token != ""):
                    target_token_id.append(target_word_to_id.get(token, target_word_to_id.get('<UNK>')))

            # put <EOS> token at the end of the chosen target sentence
            # this token suggests when to stop creating a sequence
            target_token_id.append(target_word_to_id.get('<EOS>'))

            # add each converted sentences in the final list
            source_text_id.append(source_token_id)
            target_text_id.append(target_token_id)
    return source_text_id, target_text_id


def preprocess_and_save_data(source_path, target_path, text_to_ids):
    # Preprocess

    # load original data (English, French)
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    # to the lower case
    source_text = source_text.lower()
    target_text = target_text.lower()

    # create lookup tables for English and French data
    source_vocab_to_int, source_int_to_vocab = create_mapping(source_text)
    target_vocab_to_int, target_int_to_vocab = create_mapping(target_text)

    # create list of sentences whose words are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int,
                                           target_vocab_to_int)

    # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))
