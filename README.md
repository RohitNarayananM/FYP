# FYP

## **Project Title.: Vulnerability Detection on Codebase using Bidirectional NLP**

**Team members:**

| Roll No. | Name |
| --- | --- |
| AM.EN.U4CSE20351 | Prabith GS |
| AM.EN.U4AIE20160 | Rohit Narayanan M |
| AM.EN.U4AIE20114 | Arya A |
| AM.EN.U4EAC20010 | Aneesh Nadh R |

## **Abstract**

The use of natural language processing techniques for identifying vulnerabilities in source code is becoming increasingly important in the field of cybersecurity. We propose a method using transformer technology and bidirectional NLP methods to train a model for identifying vulnerabilities in source code. We will use data from the MITRE and CWE databases to pre-train the model, allowing it to detect potential vulnerabilities in programming languages. Our approach could improve the security of software systems and prevent costly security breaches. In this project, we aim to use natural language processing (NLP) techniques to identify vulnerabilities in source code. Using data from the MITRE and Common Weakness Enumeration (CWE) databases, we will train a model using transformer technology and bidirectional NLP methods. This model will allow us to detect potential vulnerabilities in source code and help prevent security breaches.

## **Background Study**

| Title & Year | Problem | Contributions | Limitations | Open problems/Future work |
| --- | --- | --- | --- | --- |
| Limits of Machine Learning for Automatic Vulnerability Detection - 2023 | Machine learning models for vulnerability detection are often overfitting to the training data and do not generalize well to new code. Machine learning models can be fooled by semantic preserving transformations and are often unable to distinguish between vulnerable code and patched code. | The paper proposes a new methodology for benchmarking machine learning models for vulnerability detection that is more robust to overfitting and semantic preserving transformations. The paper also shows that machine learning models are unable to distinguish between vulnerable code and patched code. | The paper only evaluated machine learning models on two datasets, and it is possible that the results would be different on other datasets. The paper did not explore the use of other machine learning techniques, such as deep learning, for vulnerability detection. | Future work should explore with the use of more robust machine-learning techniques for vulnerability detection. Future work should also explore the use of machine learning to detect vulnerabilities in other types of software, such as web applications and mobile apps. |
| BBVD: A BERT-based Method for Vulnerability Detection - 2022 | Existing vulnerability detection methods are often ineffective and inefficient when dealing with large amounts of sources code.Traditional vulnerability detection methods are often based on hand-crafted features, which can be time-consuming and error-prone to create. | The paper proposes a novel BERT-based method for vulnerability detection that is able to automatically extract features from source code. The paper shows that the proposed method is effective and efficient at detecting vulnerabilities in large amounts of source code. | The paper uses BERT which is only an encoder-based method. It will not perform as well as one with an encoder-decoder model. | Future work should explore the use of the proposed method on other datasets and in other types of software. Future work should also explore the use of the proposed method to improve the accuracy of vulnerability detection. |
| Security Vulnerability Detection Using Deep Learning Natural Language Processing- 2021 | The problem that is discussed in the paper is how to detect and classify security vulnerabilities in source code using deep learning natural language processing (NLP) models. The paper argues that traditional code analysis methods are often ineffective and inefficient, and proposes to use state-of-the-art transformer-based NLP models, such as BERT, to extract contextual information from raw code and identify the type of vulnerability it contains | This paper makes three main contributions:
1) Created a large dataset of raw C/C++ code containing 123 security vulnerabilities and non-vulnerable counterparts.
2) Developed and tested deep learning models, including BERT, to identify vulnerabilities in code, showing BERT's superior performance.
3) Demonstrated effective transfer learning from written English to code, despite structural differences. | 1) The paper only focuses on C/C++ code, which may limit its applicability to other programming languages that have different syntax and semantics.
2) The paper only uses the SARD dataset, which may not be representative of real-world code and vulnerabilities. The SARD dataset also has some issues such as imbalanced classes, noisy labels, and lack of diversity 
3) The paper does not compare its deep learning models with other state-of-the-art methods for software vulnerability detection, such as VulDeePecker 2 or CodeQL. This makes it hard to evaluate the relative performance and advantages of the paper’s models. | 1) How to extend the models to other programming languages besides C/C++ and how to handle the differences in syntax and semantics among them.
2) How to improve the quality and diversity of the dataset by collecting more real-world code samples with various types of vulnerabilities and fixing methods.
3) How to compare the models with other state-of-the-art methods for software vulnerability detection, such as VulDeePecker 1 or CodeQL 2, and evaluate their relative performance and advantages. |
| Detecting software vulnerabilities using Language Models - 2023 | Current state-of-the-art deep learning models for vulnerability detection are computationally expensive, making them impractical for real-time deployment | The paper proposes a novel vulnerability detection framework called VulDetect that uses a pre-trained LLM. The paper shows that VulDetect can achieve an accuracy of up to 92.65% on benchmark datasets. The results demonstrate that the proposed VulDetect outperforms the SySeVR and VulDeBERT techniques in detecting software vulnerabilities | One limitation of VulDetect is that it is only able to detect vulnerabilities in C/C++ and Java. However, the authors believe that VulDetect could be extended to other programming languages in the future. VulDetect may not be able to keep up with the pace of new vulnerabilities. The number of software vulnerabilities is constantly increasing. It is possible that VulDetect will not be able to keep up with the pace of new vulnerabilities, and that it will miss some vulnerabilities that are newly discovered. | Use VulDetect to detect other types of security flaws. VulDetect could be used to detect other types of security flaws, such as logic errors and design flaws. VulDetect could be extended to other programming languages in the future. We could do this by training VulDetect on a dataset of known vulnerable code in other programming languages. |
| CodeT5+: Open Code Large Language Models for Code Understanding and Generation - 2023 | The paper addresses the limitations in existing large language models (LLMs) used for code understanding. There are two major limitations. First, current code-focused LLMs are inflexible in their design, using fixed structures that don't adapt well to different tasks. Second, they are pre trained with a limited set of objectives, which affects their performance on various downstream code-related tasks. | To address the limitations the paper introduces  CodeT5+, a new family of open code large language models with an encoder-decoder architecture that can flexibly operate in different modes (encoder-only, decoder-only, and encoder-decoder) to support a wide range of code understanding and generation tasks.
 | 1) The model is trained on a massive dataset of code, which means that it can be biased towards certain programming languages or styles.
2)The model is still under development and can sometimes generate incorrect or incomplete code. | CodeT5+ can be extended and improved in many ways. For example, this approach to scale the models could be applied to integrate with any open-source LLMs. For example, we can use CodeT5+ to combine with the recent StarCoder or LLaMA and utilize the different contextual representations learned from these models. |

## **Challenges**

Challenges persist in vulnerability detection: (1) Generalization hurdles due to overfitting and semantic-preserving transformations, impacting model reliability in real-world contexts. (2) Limited language diversity – extending techniques to handle distinct syntax and semantics across languages is a challenge. (3) Balancing accuracy and real-time deployment – computational demands hinder scalable and efficient detection in dynamic software environments.

1. **Deliverables of Phase I**
- Compile a diverse dataset containing vulnerable and fixed code snippets from NVD, CWE listings, and GitHub repositories.
- Develop and train a machine learning model to identify vulnerability class from vulnerable snippets, and establish an initial baseline accuracy for code vulnerability detection.
- Conduct preliminary evaluations to assess the model's performance in identifying vulnerability class from vulnerable inputs.
- The final outcome of this project will be software which will take in code snippets and give the type of vulnerability class as output.

---

1. **Assumptions/Declarations:**
- CodeT5+ is a large language model (LLM) that has been pre-trained on a massive dataset of code. It can be used for a variety of tasks, including code generation, code translation, and vulnerability detection.
- Fine-tuning CodeT5+ for vulnerability detection involves training the model on a dataset of code that contains known vulnerabilities. This allows the model to learn to identify the patterns that are associated with vulnerabilities.
- SARD and BigVuln are two large datasets of code snippets that contain known vulnerabilities. They are both publicly available and can be used for a variety of purposes, including vulnerability detection, vulnerability research, and code analysis.
- SARD dataset has over 100,000 code snippets and BigVuln dataset has over 2 million code snippets that contain known vulnerabilities. The snippets are classified by vulnerability type, and they come from a variety of sources, including open-source projects, commercial software, and academic papers.

---

## **Tools to be used**

| Software/Hardware Tools | Specifications |
| --- | --- |
| Cloud Compute Engine | • CPU with at least 8 cores and 16 threads
• GPU with at least 4GB of VRAM.
• 16GB/32GB of RAM |
| TensorFlow, PyTorch, Keras, and Scikit-learn | Deep learning frameworks for model development and training. |
| BeautifulSoup, Scrapy | Web scraping tools for collecting code snippets and vulnerability data. |
| Pandas | Data manipulation library for preprocessing and analysis. |
| Transformers | (Hugging Face)	Utilize pre-trained language models for code analysis. |

## **High Level Design**

![HighlevelDesign](https://lh7-us.googleusercontent.com/c-czgmFfEcGwUALvb1kDMMWOqDlzjiAE7yJ5M7UjjG56ElncXJOCz3yTPRoY2-h2CPzbnCyX5MF1pMSrWuNbQNY1chL0UF69nyCKMVWCB-K-ve3ZMrw-5cQM8ZICmf-WGyOG_UgLeVv59AO5hGqMTMg)