# Paperpal

 Team              : CTRL C CTRL V
 
 Track Company     : Track-1 Cactus Communications
 
 Track Name        : Paperpal - Future of Academic Writing
 
 Check Type        : Content Centric Checks and Formatting Based Checks
 
 Checks Implemented : Publication Integrity Check, AI generated Content Checker, Salami Publishing Check and Image Quality Checker.
 
 Team Members      : Himanshu Vadher, Jainik Bakshi, Jayneel Shah, Smiti Kothari
 
 
 Salient Features of Each track are as follows:
 
 1. Publication Integrity Check
 
     1.1. Typo errors are flagged and checked, by converting a pdf and preprocessing data using series of  preprocessors, autocorrects and spell checkers.
     
     1.2. Plagiarism Checker uses web scraping from Wikipedia, assumed to be the authentic source, converted to a corpus, passed through a series of preprocessors and           uses state-of-the art doc2Vec model. The model is trained and evaluated for, against our pdf/docx, uses co-sine similarity and produces results in decreasing           sorted order of similarity.

2.  AI generated Content Checker

    2.1. The state-of-the-art model, 'Roberta-base-openai-detector' is used for text-classification from HuggingFace.
    
    2.2. The pipeline is constructed and then it is tokenized and the model is ready for use.
    
    2.3. The PDF text is now tokenized and passed through the model, and it returns output and label.
    
    2.4. The confidence is given as an output, which shows what percentage of the PDF is written by AI or Human.
    
    
3.  Salami Publishing Check
   
    3.1. The author name is searched on the google scholar and every research paper, available publicly is downloaded.
   
    3.2. By far, the most complicated and useful check implemented using best preprocessing techniques, which skims the abstract, introduction and conclusion of              every research paper of an author. 
   
    3.3. This then goes through series of checks, which summarizes the text and uses Doc2Vec model for further processing.
   
    3.4. The model then gives output as similarity index with the mentioned pdf, which gives us an idea of author's work and how they are similar to each other.
    

4. Image Quality Checker

   4.1. Images are extracted from the pdf file using the python library PyMUPDF
   
   4.2. Images are then passed through a series of processing using cv2 of python, which flags the image as blurred or High resolution.
   
   4.3. By the end of this step, the user has different local drives containing high resolution and blurred images.
   
   4.4. The blurred images are now, passed through a state-of-the-art SRGAN model, which is trained on the dataset, which converts low resolution image to high             resolution image and gives output to the user.
   
   4.5. The High resolution images obtained are automatically stored in the High resolution Folder.
   

