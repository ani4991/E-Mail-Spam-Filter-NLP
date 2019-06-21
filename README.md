# E-Mail-Spam-Filter-Natural-Language-Processing


The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS          messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.                                                     
                                                                                                                                           
The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw  text.                                                                                                                                     

Tf–idf stands for "Term Frequency–Inverse Document Frequency" is a numerical statistic used to reflect how important a word is to a        document in a collection or corpus of documents.                                                                                           
TFIDF is used as a weighting factor during text search processes and text mining.                                                         
The intuition behing the TFIDF is as follows: if a word appears several times in a given document, this word might be meaningful (more      important) than other words that appeared fewer times in the same document. However, if a given word appeared several times in a given      document but also appeared many times in other documents, there is a probability that this word might be common frequent word such as 'I'  'am'..etc. (not really important or meaningful!).                                                                                         
TF: Term Frequency is used to measure the frequency of term occurrence in a document:                                                     
TF(word) = Number of times the 'word' appears in a document / Total number of terms in the document                                       
IDF: Inverse Document Frequency is used to measure how important a term is:                                                               
                                                                                                                                           
IDF(word) = log_e(Total number of documents / Number of documents with the term 'word' in it).                                             
