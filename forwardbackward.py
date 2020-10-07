import sys
import numpy as np
import math


def data_split(Data):
    all_sentences = []
    for i in Data:
        all_sentences.append(i.strip().split('\n'))
    sentence_words = [j[0].split(' ') for j in all_sentences]
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
    return list_input


def forward(list_input, A, B, Pi):
    Pi = np.reshape(Pi, (1, 9))
    all_alpha = []

    for each_sentence in list_input:
        alpha_list = []
        initial_alpha = Pi * B[:, each_sentence[0]['word']]
        alpha_list.append(initial_alpha)
        for index in range(1, len(each_sentence)):
            inner_mult = np.dot(alpha_list[index-1], A)
            alpha = B[:, each_sentence[index]['word']] * inner_mult
            alpha_list.append(alpha)
        all_alpha.append(alpha_list)

    return all_alpha


def backward(list_input, A, B, Pi):
    Pi = np.reshape(Pi, (1, 9))
    all_beta = []

    for each_sentence in list_input:
        beta_list = []
        beta_last = np.ones(len(Pi.T))
        beta_list.append(beta_last)
        i = 0
        for index in range(len(each_sentence)-1, 0, -1):
            inner_mult = B[:, each_sentence[index]['word']] * beta_list[i]
            beta = np.dot(inner_mult, A.T)
            beta_list.append(beta)
            i += 1
        beta_list.reverse()
        all_beta.append(beta_list)
    return all_beta


def predict(list_input, all_alpha, all_beta, tags_dict_ulta, words_dict_ulta):
    all_predictions = []
    accuracy = 0
    total_words = 0

    sentence_index = 0
    for each_sentence in list_input:
        total_words += len(each_sentence)
        sentence_prediction = []
        word_index = 0
        for each_dict in each_sentence:
            alpha = all_alpha[sentence_index][word_index]
            beta = all_beta[sentence_index][word_index]

            probability = alpha * beta
            prediction = np.argmax(probability)

            real_tag = each_dict['tag']
            if real_tag == prediction:
                accuracy += 1

            predict_word = words_dict_ulta[each_dict['word']]
            predict_tag = tags_dict_ulta[prediction]
            sentence_prediction.append(predict_word + '_' + predict_tag)
            word_index += 1
        all_predictions.append(sentence_prediction)
        sentence_index += 1

    accuracy = accuracy / total_words
    return all_predictions, accuracy


def log_like(all_alpha):
    avg_log_like = 0

    for alpha_sentence in all_alpha:
        last_alpha = alpha_sentence[-1]
        inner_sum = np.sum(last_alpha)
        if inner_sum > 0:
            avg_log_like += math.log(inner_sum)
    avg_log_like = avg_log_like / len(all_alpha)
    return avg_log_like


if __name__ == "__main__":
    # test_input = "fulldata/testwords.txt"
    # word_txt = "fulldata/index_to_word.txt"
    # tag_txt = "fulldata/index_to_tag.txt"
    # prior_out = 'hmmprior.txt'
    # emission_out = 'hmmemit.txt'
    # transition_out = 'hmmtrans.txt'
    # predicted_file = 'predictions.txt'
    # metric_file = 'metrics_output.txt'
    test_input = sys.argv[1]
    word_txt = sys.argv[2]
    tag_txt = sys.argv[3]
    prior_out = sys.argv[4]
    emission_out = sys.argv[5]
    transition_out = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    word_txt_open = open(word_txt, 'r')
    word_txt_read = word_txt_open.readlines()

    word_list = []
    for i in word_txt_read:
        word_list.append(i.strip())
    words_dict_ulta = dict(enumerate(word_list))
    words_dict = {v: k for k, v in words_dict_ulta.items()}


############################################################################################################################################################

    tag_txt_open = open(tag_txt, 'r')
    tag_txt_read = tag_txt_open.readlines()

    tag_list = []
    for j in tag_txt_read:
        tag_list.append(j.strip())
    tags_dict_ulta = dict(enumerate(tag_list))
    tags_dict = {v: k for k, v in tags_dict_ulta.items()}


############################################################################################################################################################

    test_open = open(test_input, 'r')
    test_read = test_open.readlines()
    sentences_use = data_split(test_read)
    list_input = train_data(words_dict, tags_dict, sentences_use)

############################################################################################################################################################

    Pi = np.loadtxt(prior_out)
    B = np.loadtxt(emission_out)
    A = np.loadtxt(transition_out)

    all_alpha = forward(list_input, A, B, Pi)
    all_beta = backward(list_input, A, B, Pi)
    all_predictions, accuracy = predict(
        list_input, all_alpha, all_beta, tags_dict_ulta, words_dict_ulta)
    avg_log_likelihood = log_like(all_alpha)

    metrics_output = 'Average Log-Likelihood: {}\nAccuracy: {}\n'.format(
        avg_log_likelihood, accuracy)
    metric_open = open(metric_file, 'w')
    metric_open.write(metrics_output)

    pred_result = ''
    for sentence_pred in all_predictions:
        for word_pred in sentence_pred:
            pred_result += word_pred + ' '
        pred_result = pred_result.strip() + '\n'
    pred_open = open(predicted_file, 'w')
    pred_open.write(pred_result)
