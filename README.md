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
