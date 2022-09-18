# Bayesian CNN Digit Classifier
An app for drawing digits which are classified with a bayesian neural network classifier.

Bayesian NN's provide us with a few nice benefits:
- Captures uncertainty
- Inherent regularisation

A forward pass through a normal neural network is deterministic and using the parameters configured during training. Whereas
Bayesian Neural Networks are non-deterministic as they swap weights for probability distributions which you sample from in a forward
pass.

Try the app [here](https://tomukmatthews-bayesian-digi-bayesian-digit-classifierapp-mohrac.streamlitapp.com).

## Run the app locally
Run `streamlit run bayesian-digit-classifier/app.py`.

## Re-train the model
Run `streamlit run bayesian-digit-classifier/training.py`.

## Re-train the model
![](https://github.com/tomukmatthews/bayesian-digit-classifier/blob/main/demo.gif)
