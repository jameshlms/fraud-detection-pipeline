# Fraud Detection Pipeline

_This is a pipeline to demonstrate proficiency of various technologies to build an end-to-end data pipeline for prediciting fraudulent credit card transactions._

Fraudulent transactions is a commonly used dataset, often used to highlight the ability to resample data to train models. While this project will evaluate resampling and training techniques to address the imbalance of classes, this project will primarily be a demonstration of technologies and skills.

> The technologies and skills that are too be demonstrated are listed below:
> - Using common Python libraries for data science and machine learning to explore data and build models.
> - Exporting ONNX model files and using ONNX runtime.
> - Using C# and .NET for API development to add variety and the ability to integrate with .NET's growing usage.

## Step-by-Step Plan
This is the ideal roadmap for this project.
1. Data Exploration with Python and Jupyter Notebooks _(In-progress)_
2. Model Training using Scikit-Learn, PyTorch, or other machine learning libraries.
3. Export the model as an ONNX model file.
4. Create a ASP.NET minimal API with appropriate endpoints.
5. Load ONNX model file into a runtime using Mircosoft.ML.

## Data Source
The data is sourced from [this Kaggle page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Necessary acknowledgements are below in the _Acknowlegdements_ section.

## Acknowledgements
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

Yann-Aël Le Borgne, Gianluca Bontempi Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook

Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé, Gianluca Bontempi Incremental learning strategies for credit cards fraud detection, IInternational Journal of Data Science and Analytics
