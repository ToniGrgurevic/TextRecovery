# Text Recovery for Corrupted Sequence Restoration

## Project Overview

This project aims to develop a machine learning model capable of restoring corrupted text sequences by predicting missing characters. The primary goal is to reconstruct text where certain characters have been randomly replaced by the # symbol.

## Team

- Toni Grgurević
- Marin Bogešić
- Machine Learning, FEUP 2024

## Dataset

The project utilizes a dataset of text sequence pairs:
- Original text sequences
- Corresponding corrupted text sequences with random # symbol replacements

## Approach

### Statistical Methods
- n-gram models
- Hidden Markov Models (HMMs)

### Deep Learning Approach
- Long Short-Term Memory (LSTM) networks for capturing long-range dependencies in sequential data

## Performance Evaluation Metrics

- Character-level accuracy
- BLEU score (sequence similarity)
- Levenshtein distance (error quantification)

## Software Components

1. **Data Processing Scripts**
   - Format and preprocess input sequences
   - Prepare data for model training

2. **Model Training Scripts**
   - Develop statistical models using:
     * hmmlearn library
     * nltk library
   - Implement LSTM-based sequence prediction models

3. **Evaluation Scripts**
   - Assess model performance
   - Calculate accuracy, BLEU score, and Levenshtein distance

## Key References

1. Jelinek, F. (1997). Statistical Methods for Speech Recognition
2. Manning, C. D., & Schütze, H. (1999). Foundations of Statistical Natural Language Processing
3. Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
4. Olah, C. (2015). "Understanding LSTM Networks"

## Project Status

Initial research and methodology development phase

## License

[To be determined]

## Contact

- Toni Grgurević
- Marin Bogešić
- Machine Learning Department, FEUP
