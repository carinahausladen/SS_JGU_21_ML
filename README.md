I will add course material, data, and code continuously during the course.

# Preparation

## Install software

### Project setup
Copy this project to your local machine. I recommend [to set up git](https://github.com/git-guides/install-git) and use the following commands (Windows: PowerShell/git bash; macOS: Terminal) to clone this project to your local machine and navigate to the project directory:
```
git clone https://github.com/hauslaca/SS_JGU_21_ML.git
cd SS_JGU_21_ML
```

### Python setup

#### 1. Install Python
In this project, we use Python version 3.9. I highly recommend using the same version for this course. You can find installation instructions for [Windows](https://www.python.org/downloads/windows/) and [macOS](https://www.python.org/downloads/macos/) online.

#### 2. Create and activate a virtual environment
If you are having multiple Python projects on your computer, you might want to avoid dependency conflicts between projects. In that case, you might set up a virtual environment for the project: navigate to your project folder and execute the following commands:

On Windows:
```
python -m venv .venv
.venv\Scripts\activate.bat
```
On macOS:
```
python -m venv .venv
source .venv/bin/activate
```

See [here](https://docs.python.org/3/tutorial/venv.html) for further information on virtual environments.

#### 3. Install python dependencies
I use several Python libraries that are not included in vanilla Python. Please install them by calling the following command from your project directory:
```
pip install -r requirements.txt
```

#### 4. Start the Jupyter environment
I use Jupyter Lab to execute interactive Python notebooks. You can start it by running:
```
jupyter lab
```

### R setup
Please find installation instructions for R/R Studio [online](https://www.rstudio.com/products/rstudio/download/).


## Tutorials
This course is designed for beginners. It is **not required** to work through the tutorials listed. However, having seen basic Python/R syntax will be helpful. My course does not cover an introduction to Python/R. Therefore, investing an hour or so in familiarizing yourself with the basics will make the course more fun!
* Python 
  * [python.org](https://www.python.org/about/gettingstarted/)
  * [coursera](https://www.coursera.org/specializations/python?utm_source=gg&utm_medium=sem&utm_campaign=06-PythonforEverybody-ROW&utm_content=06-PythonforEverybody-ROW&campaignid=6493101579&adgroupid=83560622768&device=c&keyword=python%20for%20beginners&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=506843632505&hide_mobile_promo&gclid=CjwKCAjwhOyJBhA4EiwAEcJdcZsfpYmsCKGzD5HhL1NH7j8r6Td4fHEF3ysyi_2uzWyRJlP0YeuanxoCFRkQAvD_BwE)
  * [datacamp](https://www.datacamp.com/courses/intro-to-python-for-data-science?utm_source=adwords_ppc&utm_campaignid=654038607&utm_adgroupid=50853775179&utm_device=c&utm_keyword=python%20programming&utm_matchtype=p&utm_network=g&utm_adpostion=&utm_creative=242462246308&utm_targetid=kwd-299109647448&utm_loc_interest_ms=&utm_loc_physical_ms=1003297&gclid=CjwKCAjwhOyJBhA4EiwAEcJdce_waibWFOv_kW7LMaRLdMs739KSuQV8LeivuivgSyrQsohZP_IbOhoCSnMQAvD_BwE)
* R
  * [education/rstudio](https://education.rstudio.com/learn/beginner/)
  * [cran](https://cran.r-project.org/doc/contrib/Paradis-rdebuts_en.pdf)
  * [datacamp](https://www.datacamp.com/courses/free-introduction-to-r?utm_source=adwords_ppc&utm_campaignid=12492439676&utm_adgroupid=122563401561&utm_device=c&utm_keyword=rstudio%20tutorial&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=504158801446&utm_targetid=kwd-345430083004&utm_loc_interest_ms=&utm_loc_physical_ms=1003297&gclid=CjwKCAjwhOyJBhA4EiwAEcJdcf5ciC6qhsn4qvC-Y0tFR3e2TB8szoODvfHo3aLI8xlsOwN3qFD8LxoCousQAvD_BwE)

## Readings 
I do not expect you to read all docs, papers, and books listed. However, I think that clicking through the docs/related papers helps you to structure your recall concerning the methods we will use.
* Books
  * [Natural Language Processing with Python](http://www.nltk.org/book/)
  * [Hands-on machine learning with Scikit-learn, Keras, and TensorFlow](https://books.google.ch/books?hl=de&lr=&id=HHetDwAAQBAJ&oi=fnd&pg=PP1&dq=Hands-on+machine+learning+with+Scikit-learn,+Keras,+and+TensorFlow+:+concepts,+tools,+and+techniques+to+build+intelligent+systems+Géron,+Aurélien,&ots=0LqeZsghUx&sig=vztt-zROGnWcpKpY-r0FcqLOq34#v=onepage&q=Hands-on%20machine%20learning%20with%20Scikit-learn%2C%20Keras%2C%20and%20TensorFlow%20%3A%20concepts%2C%20tools%2C%20and%20techniques%20to%20build%20intelligent%20systems%20Géron%2C%20Aurélien%2C&f=false)
  * Papers/docs are linked in the following paragraphs.

# Schedule

30.09.2021

| Time          | Session   |
| ------------- | ------------- |
| 10:00–11:30   | Elections and Divisiveness |
| 11:30–11:45   | Break |
| 11:45–13:15   | Tax Risk Management |
| 13:15–14:30   | Lunch Break |
| 14:30–16:00   | Predicting (Non) Compliance |

01.10.2021

| Time          | Session   |
| ------------- | ------------- |
| 10:00–11:30   | Ideological Direction in Judicial Opinion |
| 11:30–11:45   | Break |
| 11:45–13:15   | Charting the Type Space |

# Overview
In this course, you will learn to apply machine learning (ML) tools to data to answer research questions that are interesting for social scientists. 
This is an application-based course, focusing on R and Python as programming languages. This course is structured around projects and papers that I worked on during my time as a Ph.D. student. Topics covered are located in the fields of law and economics, behavioural, and experimental economics. Within each of the topics, we will explore relevant (ML) methods.


# I. Elections and Divisiveness 

:chart_with_upwards_trend: [Plenarprotokolle Deutscher Bundestag](https://www.bundestag.de/dokumente/protokolle/plenarprotokolle)

* Programming language: R (mainly [quanteda](https://quanteda.io))
* Methods
    * [Dfm](https://quanteda.io/reference/dfm.html), [keywords in context](https://quanteda.io/reference/kwic.html) 
    * Readability scores ([overview](https://quanteda.io/reference/textstat_readability.html), [Flesch (1948)](https://psycnet.apa.org/record/1949-01274-001))
    * [Comparing texts](https://quanteda.io/reference/textstat_simil.html) (e.g. via cosine similarity)
    * Classification
      * Naive Bayes Classifier ([Schütze et al. (2008), ch 13](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf))
      * Evaluation metrics
    * Scaling models
      * Wordscore ([Laver et al. (2003)](https://www.cambridge.org/core/journals/american-political-science-review/article/extracting-policy-positions-from-political-texts-using-words-as-data/4F4676E80A79E01EAAB88EF3F2D1B733)) 
      * Wordfish ([Slapin (2008)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-5907.2008.00338.x))
      * Wordshoal ([Lauderdale (2016)](https://www.cambridge.org/core/journals/political-analysis/article/measuring-political-positions-from-legislative-speech/35D8B53C4B7367185325C25BBE5F42B4))
    * Structural Topic Models ([Roberts (2019)](https://www.jstatsoft.org/article/view/v091i02))
    * Measuring divisiveness in political speech ([Ash (2017)](https://www.journals.uchicago.edu/doi/full/10.1086/692587?af=R&mobileUi=0))
    
#  II. Tax Risk Management
:chart_with_upwards_trend: Tax reports from German companies listed in the STOXX Europe 600.
* Programming language: Python (mostly [scikit-learn](https://scikit-learn.org/stable/))
* Methods
  * [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html)
  * Classification
    * [Multinomial Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes) 
    * [Passive Aggressive Algorithms](https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms)
    * [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#classification)
  * RCNN ([Lai et al. (2015)](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9745))
  * Clustering ([K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means))
  * Visualize high-dimensional data ([t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne))
  * Topic Models ([NFM](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf), [LDA](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda))

# III. Predicting (Non) Compliance

:chart_with_upwards_trend: chat data from a [laboratory experiment](https://www.socialscienceregistry.org/trials/5049)
* Programming language: Python (mostly [scikit-learn](https://scikit-learn.org/stable/))
* Methods
  * Feature engineering
    * Input: concatenating chat on different levels (individuals, versus groups)
    * Output: applying different thresholds to binarize the continuous dependent variable
  * Sampling ([oversampling](https://imbalanced-learn.org/stable/over_sampling.html#random-over-sampler))
  * Embeddings
    * Word2Vec ([Mikolov (2013)](https://arxiv.org/abs/1301.3781))
    * GloVe ([Pennington (2014)](https://aclanthology.org/D14-1162.pdf))
    * fastText ([Joulin (2016)](https://arxiv.org/pdf/1607.01759.pdf?fbclid=IwAR1wttEXho2gqk3BasKDuncgftN5I5lmH2TbIgvGuHxfutM3IavbateHH9A))
    * PV-DM ([Le (2014)](http://proceedings.mlr.press/v32/le14.html))
  * Classification
    * [Linear classifier (log loss) with SGD training](https://scikit-learn.org/stable/modules/sgd.html#sgd)
    * [Ridge classifier](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
    * [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting), specifically [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    * [ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
      * [bagging classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
      * [stacked generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
  * Scoring
    * [Accuracy, precision, recall, f1-score, AUC](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
    * Evaluate a model's performance within and across datasets ([Straube (2014)](https://www.frontiersin.org/articles/10.3389/fncom.2014.00043/full))
  * Sentiment analysis (dictionary based)

    
# IV. Ideological Direction in Judicial Opinion

:chart_with_upwards_trend: US Circuit Court judicial decisions
* Programming language: Python (mostly [scikit-learn](https://scikit-learn.org/stable/))
* Methods
  * Feature engineering: exploiting the specific structure of judicial texts (citations/quotations) ([Hausladen et al. 2020](https://www.sciencedirect.com/science/article/pii/S0144818819303667))
  * ML predictions as regression covariates ([Fong (2020)](https://www.cambridge.org/core/journals/political-analysis/article/machine-learning-predictions-as-regression-covariates/462A74A46A97C20A17CF640BDA72B826))
  * [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html#)

# V. Charting the Type Space

:chart_with_upwards_trend: decision data from multi-stage public good games
* Programming language: R (mostly [dtw clust](https://cran.r-project.org/web/packages/dtwclust/dtwclust.pdf))
* Methods ([Sarda-Espinosa (2017)](https://mran.microsoft.com/snapshot/2018-07-24/web/packages/dtwclust/vignettes/dtwclust.pdf))
    * distance measures (dtw, sdtw)
    * clustering algorithm (hierarchical, partitional)
    * prototypes (dba)
    * cluster evaluation (e.g. majority vote of different indices)
    

            
        
        
