# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 23:18:48 2019

@author: KARTHI
"""
import nltk

import sys
import sklearn

print('python: {}'.format(sys.version))
print('nltk: {}'.format(nltk.__version__))
print('sklearn: {}'.format(sklearn.__version__))
# nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hello students, how are you doing today?, The olympics are inspiring, and python is awesome.', 'You look great today."

print(sent_tokenize(text))

print(word_tokenize(text))

# Removing stop words - useless data
from nltk.corpus import stopwords
print(set(stopwords.words('english')))

example = 'This is some sample text,showing of stop words filtration.'

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)

filtered_sentence = [w for w in word_tokens if not w in stop_words]
"""
filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
"""
        
print(word_tokens)
print(filtered_sentence)

# Stemming words with NLTK
from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ['ride', 'riding', 'rider', 'rides']

for w in example_words:
    print(ps.stem(w))

# Stemming an entire sentence
new_text = 'when riders are riding their horses, they often think of how cowboys rode horses.'

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
    
"""
Part of Speech
1.Tagging
2.chunking
3.Entity recognition

"""
from nltk.corpus import udhr # udhr - universal declaration of human rights
print(udhr.raw('English-Latin1'))

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')
print(train_text)

# Now that we have some text, we can train the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Now lets Tokenize the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)
print(tokenized)
 
# Define a function that will tag each tokenized word with a part of speech
def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
        
process_content()

nltk.help.upenn_tagset()

# Chunking - Grouping the words into meaningfull clusters.
"""
+ = Match 1 or more
? = Match 0 or 1 repetitions
* = Match 0 or more repetitions
. = Any character except a new line

"""

# Chunking with NLTK
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# Now that we have some text, we can train the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Now let's the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Define a function that will tag each tokenized word with a part of speech
def process_content():
    try:
        for i in tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part of speech tag with a regular expression
            chunkGram = r"""chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked =  chunkParser.parse(tagged)
            
            # Draw the chunks with nltk
            chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()
   
"""
<RB.?>* = "0 or more of any tense of adverb," followed by
<VB.?>* = "0 or more of any tense of verb," followed by
<NNP>* = "one or more proper nouns," followed by
<NN>? = "zero or one singular noun."
"""        

# Chunking with NLTK
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# Now that we have some text, we can train the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Now let's the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Define a function that will tag each tokenized word with a part of speech
def process_content():
    try:
        for i in tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # combine the part of speech tag with a regular expression
            chunkGram = r"""chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked =  chunkParser.parse(tagged)
            
            # print the nltk tree
            for subtree in chunked.subtrees(filter = lambda t: t.label() == 'chunk'):
                print(subtree)
                
            # Draw the chunks with nltk
            chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()

# Chinking with nltk
# Chinking - Remove unwanted words in chunking

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# Now that we have some text, we can train the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Now let's the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Define a function that will tag each tokenized word with a part of speech
def process_content():
    try:
        for i in tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            # The main difference here is the }{, vs.the {}. this means we're removing
            # from the chink one or more verbs, preposition, determiners, or the word 'to'.
            
            # combine the part of speech tag with a regular expression
            chunkGram = r"""chunk: {<.*>+}
                                         }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked =  chunkParser.parse(tagged)
            
            # print the nltk tree
            print(chunked)
            for subtree in chunked.subtrees(filter = lambda t: t.label() == 'chunk'):
                print(subtree)
                
            # Draw the chunks with nltk
            chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()


# Entity Recognition
# Entity Recognition-finding places,loctions,things and other informations 

def process_content():
    try:
        for i in tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary = True)
           
            # Draw the chunks with nltk
            namedEnt.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()

def process_content():
    try:
        for i in tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary = False)
           
            # Draw the chunks with nltk
            namedEnt.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()












