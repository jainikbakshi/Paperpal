import os
import PyPDF2
import spacy
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fitz
import io
import cv2
import argparse
import glob
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import random
from autocorrect import Speller
import wikipedia

file = open('report.txt', mode='w+')
image = Image.open('icon.png')
st.title('Manuscript Checker')

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image, width = 100)

with col3:
    st.write(' ')

try:
  uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'])
  if uploaded_file is not None:
    path_in = uploaded_file.name
    print(path_in)
  else:
    path_in = None
    # st.warning('This is a warning', icon="⚠️")
except:
  st.error("Please make sure that you only enter a number")
  st.stop()

check_choice = st.selectbox('Which check would you like to implement?', ['Salami Check', 'Generative AI Check', 'Image Quality Check', 'Publication Integrity Check'])


# Define the CSS for the selectbox
css = """
<style>
    .st-dl {
        border: 2px red;
        color: blue;
        background-color: white;
        font-size: 20px;
    }
    .st-dl .st-dl-container {
        max-height: 300px;
    }
    .st-bi {
            border-color: red;
            color: blue;
            background-color: white;
            }
</style>
"""

# Render the CSS
st.markdown(css, unsafe_allow_html=True)


if check_choice == 'Salami Check':
   
  import en_core_web_sm
  nlp = en_core_web_sm.load()


  file_size = os.path.getsize(path_in)


  def getACA(file):
    pdfFileObj = open(file, 'rb')

    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    text = ''

    num_pages = len(pdfReader.pages)
        
    for page_num in range(num_pages):
      page = pdfReader.pages[page_num]
      page_text = page.extract_text()
      text += page_text

    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(text) 
      
    filtered_sentence = [] 
    
    for w in word_tokens: 
      if w not in stop_words: 
          filtered_sentence.append(w) 

    text=" ".join(filtered_sentence)

    pos = text.lower().find('abstract')
  
    

    pos1 = text.lower().find('keywords')
    text1 = text[pos+len('abstract'):pos1]

    pos2 = text.lower().find('conclusion')
    pos3 = text.lower().find('references')
    text2 = text[pos2+len('conclusion'):pos3]

    text3=text[:pos]

    char_str = '' .join((z for z in text3 if not z.isdigit()))

    text5 = []
    doc=nlp(char_str)
    

    list1=[]
    for ent in doc.ents:
      if ent.label_ == 'PERSON':
        text4=ent.text
        text5.append(text4)

    if(pos == file_size):
      text1=[]
      text2=[]
      text5=[]
    list1.append(text1)
    
    list1.append(text2)
    list1.append(text5)
    return list1

  vals = getACA(path_in)
  main_abs = vals[0]
  main_concl = vals[1]
  main_string = main_abs + ' ' + main_concl
  # st.write(vals)

  def download_pdfs(links):
      for i, link in enumerate(links):
          response = requests.get(link)
          new_name = f"{i+1}.pdf"
          with open(new_name, 'wb') as f:
              f.write(response.content)

  def crawl_google_scholar(author_name):
      url = f'https://scholar.google.com/scholar?q={author_name}'
      response = requests.get(url)
      soup = BeautifulSoup(response.text, 'html.parser')
      
      links = soup.find_all('a', href=True)
      pdf_links = [link['href'] for link in links if link['href'].endswith('.pdf')]
      return pdf_links

  pdf_links=[]
  for name in vals[2]:
    pdf_links += crawl_google_scholar(name)
  # st.write(pdf_links)
  download_pdfs(pdf_links)

  selected_strings = {}
  abstracts=[]
  conclusions=[]
  for i in range(1,len(pdf_links)):
    try:
      vals = getACA(str(i)+".pdf")
      abstracts.append(vals[0])
      conclusions.append(vals[1])
      selected_strings[str(i)+".pdf"] = vals[0] + ' ' + vals[1]
    except:
      continue

  model = gensim.models.doc2vec.Doc2Vec(vector_size=30, min_count=2, epochs=80)
  tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(main_string)]
  model.build_vocab(tagged_data)
  model.train(tagged_data, total_examples=model.corpus_count, epochs=80)

  data = word_tokenize(main_string)
  main_vector = model.infer_vector(data)

  selected_vectors = {}
  for i in selected_strings.keys():
    data = word_tokenize(selected_strings[i])
    selected_vectors[i] = model.infer_vector(data)

  def cosine_sim(a, b):
    sum = 0
    sum_a = 0
    sum_b = 0
    for i in range(len(a)):
      sum_a += a[i]**2
      sum_b += b[i]**2
      sum += (a[i]*b[i])

    return sum/((sum_a**(1/2))*(sum_b**(1/2)))

  cos_vals = {}
  for i in selected_vectors.keys():
    cos_vals[i] = cosine_sim(main_vector, selected_vectors[i])

  # st.write("The similarity of papers in descending order of similarity is:")
  exceed_threshold = ''
  ranked = sorted(cos_vals, key=lambda x:cos_vals[x], reverse=True)
  for i in ranked:
    if (cos_vals[i] > 0.3):
      exceed_threshold = str(i) + ' '+ str(cos_vals[i])
    # st.write(i,':', cos_vals[i])
  file.write("Abstract, Conclusion and Authors\n")
  file.write(str(vals))
  file.write("\nPDF Links of all research papers of all authors\n")
  file.write(str(pdf_links))
  file.write("\nSimilarity Matching\n")
  file.write(exceed_threshold)
  file.close()

elif check_choice == 'Generative AI Check':
  model = pipeline("text-classification", model="roberta-base-openai-detector")
  tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
  model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
  def classify_pdf(filename):
    # Open the PDF file in binary mode
    with open(filename, 'rb') as file:
        # Read the contents of the file
        reader = PyPDF2.PdfReader(file)
        text = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text()
    
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Classify the text as either human or machine-generated
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    labels = ['human', 'machine']
    label_index = torch.argmax(probabilities).item()
    label = labels[label_index]
    confidence = probabilities[:, label_index].item()
    return label, confidence

  label, confidence = classify_pdf(path_in)
  # st.write(f"PDF file: {path_in}\nLabel: {label}\nConfidence: {confidence}\n")
  file.write('Label: ' + str(label))
  file.write('Confidence: ' + str(confidence))
  file.close()

elif check_choice == 'Image Quality Check':
  pdf_file = fitz.open(path_in)
  for page_index in range(len(pdf_file)):
    page = pdf_file[page_index]
    #print(type(page))
    image_list = page.get_images()

    for image_index, img in enumerate(page.get_images(), start=1):
      xref = img[0]
      base_image = pdf_file.extract_image(xref)
      #print(base_image)
      image_bytes = base_image['image']
      image_ext = base_image['ext']
      image = Image.open(io.BytesIO(image_bytes))
      # save it to local disk
      image.save(open(f"image{page_index+1}_{image_index}.{image_ext}", "wb"))
      img = cv2.imread(f"image{page_index+1}_{image_index}.{image_ext}")
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      fm = cv2.Laplacian(gray, cv2.CV_64F).var()
      text = "Not Blurry"
      if fm < 1000:
        text = "Blurry"
        
      
      if text == "Blurry":
        image.save(open(f"Blurry/image{page_index+1}_{image_index}.{image_ext}", "wb"))
        file.write('Image is blurry' + str(fm))
        # st.write(fm)
      else:
        image.save(open(f"HighResolution/image{page_index+1}_{image_index}.{image_ext}", "wb"))
        file.write('Image is high resolution' + str(fm))
        # st.write(fm)
  SIZE = 256
  final_model = load_model('final_model.h5', compile=False)
  def plot_images(high,low,predicted):
    fig = plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 20)
    plt.imshow(high)
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 20)
    plt.imshow(low)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 20)
    plt.imshow(predicted)
    plt.plot(fig)
    st.pyplot()
  low_img = []
  path = 'Blurry/'
  files = os.listdir(path)
  for i in tqdm(files):
    try:
      img = cv2.imread(path + '/'+i,1)
      #resizing image
      img = cv2.resize(img, (SIZE, SIZE))
      img = img.astype('float32') / 255.0
      low_img.append(tf.keras.utils.img_to_array(img))
    except:
      break
  for i in range(len(low_img)):
    predicted = final_model.predict(low_img[i].reshape(1,SIZE, SIZE,3))
    image.save(open(f"HighResolution/image{page_index+1}_{image_index}.{image_ext}", "wb"))

  file.write('The blurry images have been corrected and stored in your local machine.')
  file.close()

elif check_choice == 'Publication Integrity Check':
  pdfReader = PyPDF2.PdfReader(path_in)
  num_pages = len(pdfReader.pages)
  count = 0
  text = ""
    #The while loop will read each page.
  while count < num_pages:
    pageObj = pdfReader.pages[count]
    count +=1
    text += pageObj.extract_text()

  import re

  words = re.findall(r"[\s\w]", text)
  words = "".join(words)
  words = words.lower()
  for i in range(len(words)):
    if words[i].isdigit():
      words = words[:i]+' '+words[i+1:]

  words = words.split()
  for i in words:
    if len(i)<3:
      words.remove(i)

  stop = nltk.corpus.stopwords.words('english')
  for i in stop:
    while i in words:
      words.remove(i)

  words = ' '.join(words)

  x = words.split()
  tokens = []
  for i in x:
    if len(i)>3:
      tokens.append(i)

  tags = nltk.pos_tag(tokens)

  spell = Speller()
  for i in range(len(tokens)):
    if tags[i][1] not in ['NN', 'JJ', 'NNP']:
      tokens[i] = spell(tokens[i])
  
  wiki_pages = []
  for i in tokens[:1000]:
    try:
      wiki_pages.append(wikipedia.page(i))
    except:
      continue

  model = gensim.models.doc2vec.Doc2Vec(vector_size=30, min_count=2, epochs=80)
  tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text)]
  model.build_vocab(tagged_data)
  model.train(tagged_data, total_examples=model.corpus_count, epochs=80)

  data = word_tokenize(' '.join(tokens))
  token_vector = model.infer_vector(data)

  wiki_dict = {}
  for i in wiki_pages:
    data = word_tokenize(i.content)
    wiki_dict[i.url] = model.infer_vector(data)

  def cosine_sim(a, b):
    sum = 0
    sum_a = 0
    sum_b = 0
    for i in range(len(a)):
      sum_a += a[i]**2
      sum_b += b[i]**2
      sum += (a[i]*b[i])

    return sum/((sum_a**(1/2))*(sum_b**(1/2)))

  cos_vals = {}
  for i in wiki_pages:
    cos_vals[i.url] = cosine_sim(token_vector, wiki_dict[i.url])

  st.write('The descending order of similarity index with different websites is:')
  ranked = sorted(cos_vals, key=lambda x:cos_vals[x], reverse=True)
  for i in ranked:
    st.write(i, cos_vals[i])
    file.write(str(i) + ' : ' + str(cos_vals[i]))
  file.close()
  
file1 = open("C:\\Users\\jayne\\OneDrive\\Desktop\\Mined Hackathon 2023\\report.txt", mode = 'r+')
st.download_button('Download Report', file1, file_name='checkReport.txt')
