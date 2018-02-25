import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id in range(len(test_set.get_all_Xlengths())):
        x, lengths = test_set.get_item_Xlengths(word_id)
        word_probabilities = {}
        for word, model in models.items():
            try:
                log_l = model.score(x, lengths)
                word_probabilities[word] = log_l
            except (ValueError, AttributeError):
                continue
        probabilities.append(word_probabilities)
        top_word_probabilities = sorted(word_probabilities.items(), key=lambda item: item[1], reverse=True)
        guesses.append([guess for guess, score in top_word_probabilities][0])

    #print(guesses[:10])

    return probabilities, guesses
