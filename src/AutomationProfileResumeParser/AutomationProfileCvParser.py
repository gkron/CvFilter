


#importing all required libraries

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import PyPDF2

import os
import numpy as np
from os import listdir

from os.path import isfile, join

from io import StringIO

import pandas as pd

from collections import Counter

import en_core_web_sm


nlp = en_core_web_sm.load()

from spacy.matcher import PhraseMatcher


import docx
import nltk
#Function to read resumes from the folder one by one
#cwd = os.getcwd()
#print(cwd)
#mypath= cwd+'//Resumepdfsamples'
mypath='D:/eclipse-workspace/ResumeParserUtilty/ResumeSamplesInDocsFormat' #enter your path here where you saved the resumes

onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]



#function to read resume ends

def getDocxContent(filename):
    doc = docx.Document(filename)
    fullText = ""
    for para in doc.paragraphs:
        fullText += para.text
    return fullText
    

#function that does phrase matching and builds a candidate profile

def create_profile(file):


    text = getDocxContent(file) 
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    

    #below is the csv where we have all the keywords, you can customize your own
   # cwd = os.getcwd()
    keyword_dict = pd.read_csv('D:/eclipse-workspace/ResumeParserUtilty/DataDictionary/AutomationProfileSearch.csv')
    AutomationTool = [nlp(text) for text in keyword_dict['Automation tools'].dropna(axis = 0)]

    java_words = [nlp(text) for text in keyword_dict['Java Language'].dropna(axis = 0)]

    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]

    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]

    JS_words = [nlp(text) for text in keyword_dict['JS Lanaguage'].dropna(axis = 0)]

    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]

    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]
    
    Bug_words = [nlp(text) for text in keyword_dict['Bug Tracking Tools'].dropna(axis = 0)]

    test_words = [nlp(text) for text in keyword_dict['Test Management Tool'].dropna(axis = 0)]

    Database_words = [nlp(text) for text in keyword_dict['DataBase'].dropna(axis = 0)]



    matcher = PhraseMatcher(nlp.vocab)


    matcher.add('AutoTool', None, *AutomationTool)

    matcher.add('JAVA', None, *java_words)

    matcher.add('ML', None, *ML_words)

    matcher.add('DL', None, *DL_words)

    matcher.add('JS', None, *JS_words)

    matcher.add('Python', None, *python_words)

    matcher.add('DE', None, *Data_Engineering_words)
    
    matcher.add('JIRA', None, *Bug_words)

    matcher.add('TM', None, *test_words)

    matcher.add('DB', None, *Database_words)

    doc = nlp(text)

    

    d = []  

    matches = matcher(doc)

    for match_id, start, end in matches:

        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'

        span = doc[start : end]  # get the matched slice of the doc

        d.append((rule_id, span.text))      

    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())

    

    ## convertimg string of keywords to dataframe

    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    print("-----------------------------------------------------")
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])

    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])

    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 

    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    

    base = os.path.basename(file)

    filename = os.path.splitext(base)[0]

       

    name = filename.split('_')

    name2 = name[0]

    name2 = name2.lower()

    ## converting str to dataframe

    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])

    

    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)



    return(dataf)

        

#function ends

        

#code to execute/call the above functions



final_database=pd.DataFrame()

i = 0 

while i < len(onlyfiles):

    file = onlyfiles[i]

    dat = create_profile(file)

    final_database = final_database.append(dat)

    i +=1

    print(final_database)



    

#code to count words under each category and visulaize it through Matplotlib



final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()

final_database2.reset_index(inplace = True)

final_database2.fillna(0,inplace=True)

new_data = final_database2.iloc[:,1:]

new_data.index = final_database2['Candidate Name']


#execute the below line if you want to see the candidate profile in a csv format

#sample2=new_data.to_csv('sample.csv')

plt.rcParams.update({'font.size': 10})

ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(8,16), stacked=True,color='#86bf91', zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Candidates skill Summary", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Candidate Names", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
#ax = new_data.plot.scatter(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
#ax= new_data.plot

labels = []

for j in new_data.columns:

    for i in new_data.index:

        label = str(j)+": " + str(new_data.loc[i][j])

        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):

    width = rect.get_width()

    if width > 0:

        x = rect.get_x()

        y = rect.get_y()

        height = rect.get_height()

        ax.text(x + width/2., y + height/2., label, ha='center', va='center')

plt.show()