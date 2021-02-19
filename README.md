i3systems_assignment
==============================

Case Study for i3systems interview process

## Problem Statement
> Identify the medical packages given the medical description containing details of past history, illness, diagnosis, etc.

## Data
Sample Data - 4999 rows & 3 fields.
* **Sample** - Procedures or reports for a particular case whether some procedure has happened for a case.
* **Medical Description** - Contains details about illness, diagnosis, discharge summary, procedures, etc.
* **Package** - There are 40 packages in all. Surgery is the most frequent class (1088) & Hospice is the least frequent class (6).

### Target Variables
#### CARDIOVASCULAR
Relating to the circulatory system, which comprises the heart and blood vessels. Cardiovascular diseases are conditions that affect the heart and blood vessels and include arteriosclerosis, coronary artery disease, heart valve disease, arrhythmia, heart failure, hypertension, orthostatic hypotension, shock, endocarditis, diseases of the aorta and its branches, disorders of the peripheral vascular system, and congenital heart disease.

#### ORTHOPEDIC
Concerned with the correction or prevention of deformities, disorders, or injuries of the skeleton and associated structures (such as tendons and ligaments)

#### RADIOLOGY
Dealing with X-rays and other high-energy radiation, especially the use of such radiation for the diagnosis and treatment of disease.

#### GASTROENTROLOGY
The branch of medicine which deals with disorders of the stomach and intestines.

#### NEUROLOGY
The branch of medicine or biology that deals with the anatomy, functions, and organic disorders of nerves and the nervous system.

#### SOAP Note
SOAP note – an acronym for *Subjective, Objective, Assessment and Plan* – is the most common method of documentation used by providers to input notes into patients' medical records.

#### GYNECOLOGY
The branch of physiology and medicine which deals with the functions and diseases specific to women and girls, especially those affecting the reproductive system.

#### UROLOGY
Urology is a part of health care that deals with diseases of the male and female urinary tract (kidneys, ureters, bladder and urethra). It also deals with the male organs that are able to make babies (penis, testes, scrotum, prostate, etc.).

#### DISCHARGE SUMMARY
Hospital discharge summaries serve as the primary documents communicating a patient's care plan to the post-hospital care team. Often, the discharge summary is the only form of communication that accompanies the patient to the next setting of care.

#### Otolaryngology
An otolaryngologist is often called an ear, nose, and throat doctor, or an ENT for short.

#### NEUROSURGERY
Surgery performed on the nervous system, especially the brain and spinal cord.

#### ONCOLOGY
The study and treatment of tumours.

#### OPTHALOMOLOGY
The branch of medicine concerned with the study and treatment of disorders and diseases of the eye.

#### NEPHROLOGY
The branch of medicine that deals with the physiology and diseases of the kidneys.

#### PEDIATRICS
Paediatrics is the area of medicine that is concerned with the treatment of children's illnesses.

#### PAIN MANAGEMENT
Pain management, pain medicine, pain control or algiatry, is a branch of medicine that uses an interdisciplinary approach for easing the suffering and improving the quality of life of those living with chronic pain.

#### PSYCHIATRY
The study and treatment of mental illness, emotional disturbance, and abnormal behaviour.

#### PODIATRY
The treatment of the feet and their ailments

#### DERMATOLOGY
Dermatology is the science that is concerned with the diagnosis and treatment of diseases of the skin, hair and nails.

#### COSMETIC / PLASTIC SURGERY
The process of reconstructing or repairing parts of the body by the transfer of tissue, either in the treatment of injury or for cosmetic reasons.

#### DENTISTRY
The treatment of diseases and other conditions that affect the teeth and gums, especially the repair and extraction of teeth and the insertion of artificial ones.

#### PHSYICAL MEDICINE - REHAB
Physical medicine and rehabilitation, also known as physiatry, is a branch of medicine that aims to enhance and restore functional ability and quality of life to people with physical impairments or disabilities.

#### SLEEP MEDICINE
Sleep medicine is a medical specialty or subspecialty devoted to the diagnosis and therapy of sleep disturbances and disorders.

#### ENDOCRINOLOGY
The branch of physiology and medicine concerned with endocrine glands and hormones.

#### BARIATRICS
The branch of medicine that deals with the study and treatment of obesity.

#### CHIROPRATIC
A system of complementary medicine based on the diagnosis and manipulative treatment of misalignments of the joints, especially those of the spinal column, which are believed to cause other disorders by affecting the nerves, muscles, and organs.

#### RHEUMATOLOGY
The study of rheumatism, arthritis, and other disorders of the joints, muscles, and ligaments.

#### AUTOPSY
Postmortem

#### HOSPICE
Hospice care is a type of health care that focuses on the palliation of a terminally ill patient's pain and symptoms and attending to their emotional and spiritual needs at the end of life.

### Possibly Overlapping Categories
* **NEUROSURGERY** could be confused with SURGERY & NEUROLOGY fields
* **OPTHALOMOLOGY** can be part of ENT
* **NEPHROLOGY** deals with kidneys & so does UROLOGY
* **PAIN MANAGEMENT** could have an overlap of multiple categories
* **CHIROPRATIC** could be confused with Neurology
* **RHEUMATOLOGY** - Deals with joints & muscles, which could overlap with many above fields

## Baselines
### Random Guessing
There are 40 target classes.
> Random Guess Accuracy = 1/40 = 0.025

### Most Frequent Class
We could simply predict the most occuring class & that could serve as baseline. In our case, *Surgery* is the class with maximum number of rows - 1088.
> Accuracy = 1088/4966 = 0.21

## Method
On observing the data, it becomes clear that the classification can be done just based on few keywords which are specific to each set. Out of the 40 categories only 5-6 categories could have overlapping content. So approaching this is a **Bag of Words** model.

**TF-IDF** - Used for keyword feature extraction
**Multinomial Naive Bayes** -
### Processing
Since, using a bag of words model. I did the following text processing -

**LowerCasing** - Lowercase every word to reduce count of unique words

**Stopwords Removal** - Extracted additional medical stopwords from [here](https://cs.stanford.edu/people/sonal/gupta14jamia_supl.pdf) and [here](https://github.com/kavgan/clinical-concepts)

### Modelling
Naive Bayes is based on Bayes’ theorem. Naive means that features in the dataset are mutually independent i.e occurrence of one feature does not affect the probability of occurrence of the other feature.

* On smaller datasets, outperforms more powerful techniques
* Robust
* Fast & Accurate
* Performs well in text classification problem

Multinomial Naïve Bayes considers a feature vector where a given term represents its frequency.

## How to run the project
Since the categories distribution is very unstable. I have used **StratifiedKFold** as cross-validation strategy.

Run the training script inside `src` directory to train & save models
```
python train.py
```

## Project Organization
```
    ├── LICENSE
    ├── README.md          <- Project Description.
    ├── data
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── pickles            <- Stores pickles
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Stores exploratory notebooks
    │
    ├── src                <- Source code for use in this project.
    │   ├── process.py     <- Processing Class
    └── ├── train.py       <- Trains & Saves model
```