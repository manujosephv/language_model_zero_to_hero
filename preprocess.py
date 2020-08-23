import nltk


def pad_tokens(sentence_tokens, order):
    return ["<s>"] + sentence_tokens + ["</s>"]
    # return nltk.lm.preprocessing.pad_both_ends(sentence_tokens, n=order)


def clean_sentence(sentence):
    return sentence.replace(".", "").replace("\n", " ").replace("\r", "").strip()


def tokenize_to_lower_sentence(sentence, remove_punctuation):
    sentence_tokens = []
    for word in nltk.word_tokenize(sentence):
        if remove_punctuation:
            if word.isalnum():
                sentence_tokens.append(word.lower())
        else:
            sentence_tokens.append(word.lower())
    return sentence_tokens


def split_to_sentences(text):
    return nltk.sent_tokenize(text)


def create_n_grams(tokens, order):
    return nltk.ngrams(tokens, n=order)


def lm_preprocessing_pipeline(text, order, remove_punctuation=False):
    sentences = split_to_sentences(text)
    padded_sentence = []
    n_grams = []
    for sentence in sentences:
        sentence = clean_sentence(sentence)
        sentence_tokens = tokenize_to_lower_sentence(sentence, remove_punctuation)
        sentence_tokens = list(pad_tokens(sentence_tokens, order))
        n_grams.append(create_n_grams(sentence_tokens, order))
        padded_sentence += sentence_tokens
    return n_grams, padded_sentence

def create_every_grams(tokens, order):
    return nltk.everygrams(tokens, max_len=order)

def everygram_lm_preprocessing_pipeline(text, order, remove_punctuation=False):
    sentences = split_to_sentences(text)
    padded_sentence = []
    every_grams = []
    for sentence in sentences:
        sentence = clean_sentence(sentence)
        sentence_tokens = tokenize_to_lower_sentence(sentence, remove_punctuation)
        sentence_tokens = list(pad_tokens(sentence_tokens, order))
        every_grams.append(create_every_grams(sentence_tokens, order))
        padded_sentence += sentence_tokens
    return every_grams, padded_sentence

def everygram_lm_preprocessing_pipeline_w_sent(sentences, order, remove_punctuation=False):
    padded_sentence = []
    every_grams = []
    for sentence in sentences:
        # sentence = clean_sentence(sentence)
        sentence_tokens = tokenize_to_lower_sentence(sentence, remove_punctuation)
        sentence_tokens = list(pad_tokens(sentence_tokens, order))
        every_grams.append(create_every_grams(sentence_tokens, order))
        padded_sentence += sentence_tokens
    return every_grams, padded_sentence



#import contractions list and remove it in the next step
# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"thx"   : "thanks"
}

# def remove_contractions(text):
#     return contractions[text.lower()] if text.lower() in contractions.keys() else text
