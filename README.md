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
 
 
 
 
 The images of website are as follows:
![WhatsApp Image 2023-03-05 at 12 42 16 PM (1)](https://user-images.githubusercontent.com/94166841/222947846-34aadba1-a882-4e5a-8324-ae65e1d51e20.jpeg)

The report of Image Quality Checker is shown as below:
![WhatsApp Image 2023-03-05 at 1 08 02 PM](https://user-images.githubusercontent.com/94166841/222948137-b4093e03-633b-4ce1-871b-9b3cb30fbb18.jpeg)

The running of epoch of AI Generated Checker is shown as below:

![WhatsApp Image 2023-03-05 at 1 10 44 PM (1)](https://user-images.githubusercontent.com/94166841/222948254-ab467644-20c7-4394-9922-ffb2786244c7.jpeg)

The report of AI Generated Checker is shown as below:

![WhatsApp Image 2023-03-05 at 1 12 16 PM](https://user-images.githubusercontent.com/94166841/222948362-341554fc-3f29-4b1c-a110-6c170722e3c6.jpeg)

The ruuning of epoch of Salami Check is shown as below:

![WhatsApp Image 2023-03-05 at 1 14 59 PM (1)](https://user-images.githubusercontent.com/94166841/222948437-6eb799b1-558c-49e0-a6a1-dcf27e92f1e8.jpeg)

The report of Salami Check is shown as below:
![WhatsApp Image 2023-03-05 at 1 21 29 PM](https://user-images.githubusercontent.com/94166841/222948618-9a153d9c-68ec-41bf-82f9-ca1b3e076908.jpeg)

The epoch of Publication Integrity is shown as below:

![WhatsApp Image 2023-03-05 at 1 25 25 PM](https://user-images.githubusercontent.com/94166841/222948795-35455579-1305-4fd8-8b6b-94d900659bd8.jpeg)
