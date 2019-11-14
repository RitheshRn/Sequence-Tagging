#! /usr/bin/python

import sys
from collections import defaultdict
import math


"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(self, corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in range(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()
        self.rare = []
        self.tag_count = defaultdict(int)
        self.emission = defaultdict(list)
        self.t_bigram = defaultdict(int)
        self.t_trigram = defaultdict(int)
        self.transition = defaultdict(float)
        self.list_of_sentences = []
        self.list_of_tags = []
#         self.rare_words_list = []

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(self, corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in range(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:            
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

                
    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                self.tag_count[ne_tag] += count
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count
#         print(self.ngram_counts)

                
    def handle_rare(self):
        rare_word_counts = defaultdict(int)
        for word, ne_tag in self.emission_counts:            
            rare_word_counts[word] += self.emission_counts[(word, ne_tag)]
        self.rare = [i for i in rare_word_counts if rare_word_counts[i]<5 ]
        non_rare = [i for i in rare_word_counts if rare_word_counts[i] >= 5 ]
        with open('rare_words.txt', 'w') as f:
            for item in self.rare:
                f.write('%s\n' % str(item))
                
        with open('non_rare_words.txt', 'w') as f:
            for item in non_rare:
                f.write('%s\n' % str(item))
                
                
    def read_rare(self):
        with open("rare_words.txt", "r") as f:
            self.rare_words_list = f.readlines()
        self.rare_words_list = [i[:-1] for i in self.rare_words_list]
        
    def emission_parameters(self):
        for word, ne_tag in self.emission_counts:
#             if ne_tag != 'I-G_RARE_E' or ne_tag != "I-G_UPPER_E"or ne_tag != "G_UPPER_E":
            self.emission[word].append((ne_tag, self.emission_counts[(word, ne_tag)] / self.tag_count[ne_tag])) 
    
#         print(self.emission['BACKGROUND'])
        with open('emission.out','w') as f:
            for i in self.emission:
                f.write(str((i, self.emission[i]))+"\n")
        
    def transition_papameters(self):
        for ngram in self.ngram_counts[1]:
            self.t_bigram[ngram] = self.ngram_counts[1][ngram]
        for ngram in self.ngram_counts[2]:
            self.t_trigram[ngram] = self.ngram_counts[2][ngram]
        for ngram in self.ngram_counts[2]:
            self.transition[ngram] = self.ngram_counts[2][ngram] / self.ngram_counts[1][(ngram[0], ngram[1])]
        with open('transition.out','w') as f:
            for i in self.transition:
                f.write(str((i, self.transition[i]))+"\n")

        
    def Viterbi_algorithm(self):
        S = defaultdict(list)
        S[-1] = ['*']
        S[0] = ['*']
        
        
        
        # Assume max sentence length is 100 i.e. n = 100
        for i in range(1, 500):
            S[i] = [ 'O', 'I-GENE']
            
            
        for sentence in self.list_of_sentences:
            pi = defaultdict(float)
            pi[(0, '*', '*')] = 1.0
            optimal_tag_sequence = []
            ys = [None]*len(sentence)
            bp = {}
            
            for k in range(1, len(sentence)+1):
                
#                 global_max_tag = None
                
                for u in S[k-1]:
                    
                    for v in S[k]:       
                        for w in S[k-2]:
                                                        
#                             print(w)
                            temp_emission = 0
#                             print(sentence[k-1])
#                             print(self.emission[sentence[k-1]])
#                             try:
#                                 for tag, vval in self.emission[sentence[k-1]]:
# #                                     print("for loop", tag)
#                                     if tag == v:
# #                                         print("enter for loop: ", tag)
#                                         temp_emission = vval
#                             except:
#                                 print('________________________________________')
#                                 for tag, vval in self.emission['_RARE_']:
#                                     if tag == v:
#                                         temp_emission = vval

                            if sentence[k-1] in self.emission:
                                for tag, vval in self.emission[sentence[k-1]]:
#                                     print("for loop", tag)
                                    if tag == v:
#                                         print("enter for loop: ", tag)
                                        temp_emission = vval
                            else:
#                                 print('________________________________________')
                                for tag, vval in self.emission['_RARE_']:
                                    if tag == v:
                                        temp_emission = vval
                                
                                
                            
                            if (k-1, w, u) in pi:
                                temp = pi[(k-1, w, u)]*self.transition[(w, u, v)]*temp_emission
                            else:
                                temp = self.transition[(w, u, v)]*temp_emission
                                
                            if (k, u, v) not in pi:
                                pi[(k, u, v)] = temp
                                bp[(k, u, v)] = w
                                
                            else:
                                if temp > pi[(k, u, v)]:
                                    pi[(k, u, v)] = temp
                                    bp[(k, u, v)] = w
#                                     global_max_tag = w
#                                 else:
#                                     pi[(k, u, v)] = temp
#                                     bp[(k, u, v)] = w
#                             except:
#                                 pi[(k, u, v)] = temp
#                                 bp[(k, u, v)] = w
#                                 global_max_tag = w
#             print(pi)
#             

                                
#                         optimal_tag_sequence.append(global_max_tag)
            
            max_val = -1
#                 best_u = None
#                 best_v = None
            for u in S[len(sentence)-2]:
                for v in S[len(sentence)-1]:

                    temp1 = pi[(len(sentence), u, v)] * self.transition[(u, v, 'STOP')]
                    if temp1 > max_val:
                        max_val = temp1
                        ys[len(sentence)-2] = u
                        ys[len(sentence)-1] = v

            for xz in range(len(sentence)-2, 0, -1):
           
                ys[xz-1] = bp[(xz+2, ys[xz], ys[xz+1])]
#             print(ys)
#             print(pi)
#             import sys
#             sys.exit()
#             optimal_tag_sequence.append(best_u)
#             optimal_tag_sequence.append(best_v)
                
            self.list_of_tags.append(ys)  
           
            
      
        
    """
    def read_dev(self):
        with open('gene.dev', 'r') as f:
            lines = f.readlines()
        lines = [i[:-1] for i in lines]
        for idx, line in enumerate(lines):
            if len(line) > 0:
                try:
                    temp = sorted(self.emission[line], key=lambda x: x[1], reverse=True)[0][0]
                except:
#                     if line.isdigit():
#                         temp = sorted(self.emission['_NUMERIC_'], key=lambda x: x[1], reverse=True)[0][0]
#                     elif line.isupper():
#                         temp = sorted(self.emission['_UPPER_'], key=lambda x: x[1], reverse=True)[0][0]
#                     elif line in "~!@#$%^&*()__+={}[]|.;":
#                         temp = sorted(self.emission['_SPL_'], key=lambda x: x[1], reverse=True)[0][0]
#                     else:
                    temp = sorted(self.emission['_RARE_'], key=lambda x: x[1], reverse=True)[0][0]
                    
                lines[idx] = line + " " + temp + "\n"
            else:
                lines[idx] = '\n'
                
        with open('gene_dev.p1_NUSR.out','w') as f:
            for i in lines:
                f.write(i)
        """
    def read_sentences(self):
        with open('gene.dev', 'r') as f:
            lines = f.readlines()
        lines = [i[:-1] for i in lines]
        sentence = []
        for idx, line in enumerate(lines):
            if len(line) > 0:
                sentence.append(line)
            else:
                self.list_of_sentences.append(sentence)
                sentence = []
        with open('list_of_sentences.out','w') as f:
            for i in self.list_of_sentences:
                f.write(str(i)+"\n")
        
    def write_Viterbi_algorithm(self):
        lines = []
        for si, sent in enumerate(self.list_of_sentences):
            for i in range(len(sent)):
                lines.append(str(self.list_of_sentences[si][i]) + " " + str(self.list_of_tags[si][i]) + "\n")
            lines.append("\n")
                
        with open('V_1.out','w') as f:
            for i in lines:
                f.write(i)
        

                
            
            
        
        

        


def usage():
    print ("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """)

if __name__ == "__main__":

#     if len(sys.argv)!=2: # Expect exactly one argument: the training data file
#         usage()
#         sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    """
    """
    # Initialize a trigram counter
    counter = Hmm(3)
    counter.train(input)
    counter.write_counts(sys.stdout)
    
    """
    """
    
#     counter = Hmm(3)
#     counter.read_counts(input)
#     counter.emission_parameters()
#     counter.transition_papameters()
#     counter.read_dev()
    
    """
    """
    
#     counter = Hmm(3)
#     counter.read_counts(input)
#     counter.emission_parameters()
#     counter.transition_papameters()
#     counter.read_sentences()
#     counter.Viterbi_algorithm()
#     counter.write_Viterbi_algorithm()
    """
    """
    
    # Read rare_words file
#     counter.read_rare()
    
    
    
    # Collect counts
#     counter.train(input)
    
#     # Handle _RARE_
#     counter.handle_rare()
    

    
    # Write the counts
#     counter.write_counts(sys.stdout)

    # read counts
#     counter.read_counts(input)
    
    # Emission parameters
#     counter.emission_parameters()
    
    # Read dev
#     counter.read_dev()
    
    
