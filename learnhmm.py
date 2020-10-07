import sys
import numpy as np


def data_split(Data):
    all_sentences = []
    for i in Data:
        all_sentences.append(i.strip().split('\n'))
    sentence_words = [j[0].split(' ') for j in all_sentences]
    # print(sentence_words)
    return sentence_words


def train_data(words_dict, tags_dict, Data):
    list_input = []
    for i in Data:
        list_line = []
        for each_word in i:
            dict_word = {}
            temp = each_word.split('_')
            word_index = words_dict[temp[0]]
            tag_index = tags_dict[temp[1]]
            dict_word['word'] = word_index
            dict_word['tag'] = tag_index
            list_line.append(dict_word)
        list_input.append(list_line)
    # print(list_input[-1])
    # print(list_input[-1])
    return list_input


def init_prob(list_input, tags_dict):

    Pi = np.ones(len(tags_dict))
    for each_sentence in list_input:
        dict_1 = each_sentence[0]
        tag = dict_1['tag']
        Pi[tag] += 1
    Pi = Pi/(len(list_input)+len(tags_dict))
    return Pi


def transit_prob(list_input, tags_dict):
    A = np.ones((len(tags_dict), len(tags_dict)))

    for each_sentence in list_input:
        for i in range(len(each_sentence)-1):
            current_dict = each_sentence[i]
            next_dict = each_sentence[i+1]
            current_tag = current_dict['tag']
            next_tag = next_dict['tag']
            A[current_tag][next_tag] += 1

    for row_index in range(len(A)):
        row_sum = np.sum(A[row_index])
        for col_index in range(len(A[row_index])):
            A[row_index][col_index] /= row_sum
    return A


def emit_prob(words_dict, tags_dict):
    B = np.ones((len(tags_dict), len(words_dict)))

    for each_sentence in list_input:
        for i in range(len(each_sentence)):
            current_dict = each_sentence[i]
            current_word = current_dict['word']
            current_tag = current_dict['tag']
            B[current_tag][current_word] += 1

    for row_index in range(len(B)):
        row_sum = np.sum(B[row_index])
        for col_index in range(len(B[row_index])):
            B[row_index][col_index] /= row_sum
    return B


if __name__ == "__main__":
    # train_input = "fulldata/trainwords.txt"
    # word_txt = "fulldata/index_to_word.txt"
    # tag_txt = "fulldata/index_to_tag.txt"
    # prior_out = 'hmmprior.txt'
    # emission_out = 'hmmemit.txt'
    # transition_out = 'hmmtrans.txt'
    train_input = sys.argv[1]
    word_txt = sys.argv[2]
    tag_txt = sys.argv[3]
    prior_out = sys.argv[4]
    emission_out = sys.argv[5]
    transition_out = sys.argv[6]

############################################################################################################################################################

    word_txt_open = open(word_txt, 'r')
    word_txt_read = word_txt_open.readlines()

    word_list = []
    for i in word_txt_read:
        word_list.append(i.strip())
    words_dict_ulta = dict(enumerate(word_list))
    words_dict = {v: k for k, v in words_dict_ulta.items()}
    # print(words_dict)


############################################################################################################################################################

    tag_txt_open = open(tag_txt, 'r')
    tag_txt_read = tag_txt_open.readlines()

    tag_list = []
    for j in tag_txt_read:
        tag_list.append(j.strip())
    tags_dict_ulta = dict(enumerate(tag_list))
    tags_dict = {v: k for k, v in tags_dict_ulta.items()}
    # print(tags_dict)


############################################################################################################################################################

    train_open = open(train_input, 'r')
    train_read = train_open.readlines()

    sentences_use = data_split(train_read)
    list_input = train_data(words_dict, tags_dict, sentences_use)

    Pi = init_prob(list_input, tags_dict)
    A = transit_prob(list_input, tags_dict)
    B = emit_prob(words_dict, tags_dict)

    np.savetxt(prior_out, Pi)
    np.savetxt(emission_out, B)
    np.savetxt(transition_out, A)
