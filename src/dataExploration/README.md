# Info about the data

## General info about the data set

### Class distribution

Data set is slightly unbalanced approx. 61.616% of people didn't survive.

### Duplicated values
I don't have duplicated data in the dataset 

### Missing values
There are several missing values on Age column and the majority of Cabin values are missing, also I have 2 missing rows on the Embarked column.

## Statistics about features

### Numeric features

* Age : 
    - 50% of the people were between 20.12 yo and 38 yo.
    - I suspect there may be some outliers on the max age.

* Cabin : 
    - There is a strong correlation between the stateroom floor and Survived column

* Fare :
    - I suspect there may be some outliers on the max fare. There is also a possible inconsistency on the min value (min = 0).

* SibSp:
    - I suspect there may be some outliers on the max SibSp, it could also be an error if there aren't at least 7 more passengers with the same SibSp.

* Parch:
    - I suspect there may be some outliers on the max Parch

### Categorical features

* Sex : 
    - Approx. 72.88% of women survived
    - Approx. 19.45% of men survived

* Pclass :
    - Approx. 24.65% of 3rd class voyagers survived
    - Approx. 48.39% of 2nd class voyagers survived
    - Approx. 63.68% of 1st class voyagers survived

* Embarked :
    - Approx. 34.01% of voyagers that departure from Southampton survived.
    - Approx. 56.29% of voyagers that departure from Cherbourg survived.
    - Approx. 38.75% of voyagers that departure from Queenstownsurvived.


## Outliers 

### Age
    Q1 = 20.12, Q3 = 38
    IQR = 17.88
    IL = 20.12 - 1.5*17.88 = -6.7 
    SL = 38 + 1.5*17.88 = 64.82 

    I conclude that there are superior outliers.

### Fare
    Q1 = 7.91, Q3 = 31
    IQR = 23.09
    IL = 7.91 - 1.5*23.09 = -26...
    SL = 31 + 1.5*23.09 = 65.63

    I conclude that there are superior outliers.

### SibSp
    Q1 = 0, Q3 = 1
    IQR = 1
    IL = 0 - 1.5 * 1 = -...
    SL = 1 + 1.5 * 1 = 2.5

    I conclude that there are superior outliers.

### Parch

    Q1 = 0, Q3 = 0
    IQR = 0
    IL = 0
    SL = 0

    I conclude that there are superior outliers.


## Procedure with missing data

The first approach is going to be:

1. Fill up the missing Age values with the median Age value as the distribution has some outliers, on later versions I will evaluate the use of another imputation method

2. As approximately 72% of the people departed from Southampton I will suppose the 2 missing values of Embarked are Southampton

3. As I want to ship a first approach asap, I will delete the Cabin column. On a more advanced step of the project I will try to exploit this feature.

## How am I treating outliers

As the outliners don't look as errors I will maintain them.

## How am I treating errors.

As it doesn't look like there are a lot of errors on data, in case there are, I will remove the rows containing errors.

