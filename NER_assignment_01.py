#!/usr/bin/env python
# coding: utf-8

# In[1]:


def read_data(datafile):
    """
    read in text file and return list a dataset
    """
    with open(datafile, "r") as infile:
        ner_dataset =[]
        ner_word_list = []
        for line in infile:   
            line=line.rstrip()
            # split line items that are separated by tab
            lineitems = line.split("\t")

            # if line is a space, indicates start of new tweet
            if line == '':
                ner_dataset.append(ner_word_list)
                ner_word_list =[]
            else:
                word = lineitems[0]
                state = lineitems[1]  
            item_word_list = [ word, state]
            ner_word_list.append(item_word_list)
        return (ner_dataset)


# In[2]:


#read the data and assign to dataset

dataset = read_data("NER-Dataset-Train.txt")


# In[3]:


#Step1: Find states.

states = set()

for sentence in dataset:
    for word,state in sentence:
        states.add(state)

states = list(states)
print(states)


# In[4]:


#see the dataset
dataset


# In[5]:


# Step2: Calculate Start probability (π).

total_sentences = len(dataset)

start_prob = {state: 0 for state in states}

# Count the frequency of each state at the beginning of a sentence
for sentence in dataset:
    if len(sentence) > 0:
        first_state = sentence[0][1]  # Get the NER tag of the first word in the sentence
        start_prob[first_state] += 1

start_prob = {state: count / total_sentences for state, count in start_prob.items()}

print(start_prob)


# In[6]:


#Step3: Calculate transition probability (A)

transition_count = {state: {state: 0 for state in states} for state in states}

# Count the frequency of transitions between states
for sentence in dataset:
    for i in range(len(sentence) - 1):
        current_state = sentence[i][1]
        next_state = sentence[i + 1][1]
        transition_count[current_state][next_state] += 1

transition_prob = {}

for state in transition_count:
    total_transitions = sum(transition_count[state].values())
    transition_prob[state] = {next_state: count / total_transitions for next_state, count in transition_count[state].items()}

print(transition_prob)


# In[7]:


#Step4: Calculate emission probability (B)

word_count = {state: {} for state in states}
state_count = {state: 0 for state in states}

# Count the frequency of each word emitted by each state
for sentence in dataset:
    for word, state in sentence:
        if word not in word_count[state]:
            word_count[state][word] = 0
        word_count[state][word] += 1
        state_count[state] += 1

emission_prob = {}

for state in word_count:
    emission_prob[state] = {word: count / state_count[state] for word, count in word_count[state].items()}

print(emission_prob)


# In[8]:


# First order markov assumption (Bigram) where current word NER tag is based on the previous and current words

smoothing_factor = 1  # Laplace smoothing factor

transition_count = {state: {state: 0 for state in states} for state in states}
word_count = {state: {} for state in states}
state_count = {state: 0 for state in states}

# Count the frequency of transitions and emissions
for sentence in dataset:
    for i in range(len(sentence)):
        word, state = sentence[i]
        state_count[state] += 1

        if i > 0:
            prev_word, prev_state = sentence[i-1]
            transition_count[prev_state][state] += 1

        if word not in word_count[state]:
            word_count[state][word] = 0
        word_count[state][word] += 1

transition_prob = {}
emission_prob = {}

# Calculate transition probabilities with Laplace smoothing
for state in transition_count:
    total_transitions = sum(transition_count[state].values())
    total_transitions_with_smoothing = total_transitions + smoothing_factor * len(states)
    transition_prob[state] = {next_state: (count + smoothing_factor) / total_transitions_with_smoothing for next_state, count in transition_count[state].items()}

# Calculate emission probabilities with Laplace smoothing
for state in word_count:
    total_emissions = state_count[state]
    total_emissions_with_smoothing = total_emissions + smoothing_factor * len(word_count[state])
    emission_prob[state] = {word: (count + smoothing_factor) / total_emissions_with_smoothing for word, count in word_count[state].items()}

   
print("Transition probabilities (Bigram with Laplace smoothing):")
print(transition_prob)
print("----*****----")
print("Emission probabilities (Bigram with Laplace smoothing):")
print(emission_prob)


# In[9]:


'''
After calculating all these parameters apply these parameters to the Viterbi algorithm 
and test sentences as an observation to find named entities
'''
def viterbi(sentence, states, transition_prob, emission_prob, start_prob):

    # Initialization
    V = {}  # V[t][state] = the probability of the most likely sequence of states ending in state at time t
    B = {}  # B[t][state] = the previous state in the most likely sequence ending in state at time t

    for state in states:
        if emission_prob[state] and isinstance(emission_prob[state], dict):
            emission_prob_value = emission_prob[state].get(sentence[0], 0)
        else:
            emission_prob_value = 0
        V[(0, state)] = start_prob[state] * emission_prob_value
        B[(0, state)] = 'START'

    # Recursion
    for t in range(1, len(sentence)):
        for state in states:
            max_score = -1
            prev_state_max = None
            for prev_state in states:
                if prev_state in transition_prob and state in transition_prob[prev_state]:
                    transition_prob_value = transition_prob[prev_state][state]
                else:
                    transition_prob_value = 0
                if emission_prob[state] and isinstance(emission_prob[state], dict):
                    emission_prob_value = emission_prob[state].get(sentence[t], 0)
                else:
                    emission_prob_value = 0
                if isinstance(transition_prob_value, (int, float)) and isinstance(emission_prob_value, (int, float)):
                    score = V[(t - 1, prev_state)] * transition_prob_value * emission_prob_value
                    if score > max_score:
                        max_score = score
                        prev_state_max = prev_state
            V[(t, state)] = max_score
            B[(t, state)] = prev_state_max

    # Termination
    best_path_prob = max(V[(len(sentence) - 1, state)] for state in states)
    best_path = []

    # Backtracking
    best_state = max(states, key=lambda state: V[(len(sentence) - 1, state)])
    best_path.append(best_state)

    for t in range(len(sentence) - 2, -1, -1):
        if (t + 1, best_state) in B:
            best_state = B[(t + 1, best_state)]
        else:
            best_state = '0'
        best_path.append(best_state)

    best_path.reverse()

    return best_path



# Testing
#sentence = [item[0] for item in dataset[7]]
sentence = ["i'm", 'off', 'to', 'bed', '!', 'tomorrow',"i'll","go",'to','Nijverdal','and', 'meet','@ElineEpica'
            ,'there',',','on','her','sweeeet16','#partyyy']
result = viterbi(sentence, states, transition_prob, emission_prob, start_prob)
print(result)


# In[12]:


smoothing_factor = 1  # Laplace smoothing factor

transition_count = {state: {state: {state: 0 for state in states} for state in states} for state in states}
word_count = {state: {} for state in states}
state_count = {state: 0 for state in states}

# Count the frequency of transitions and emissions
for sentence in dataset:
    for i in range(len(sentence)):
        word, state = sentence[i]
        state_count[state] += 1

        if i > 1:
            prev_word, prev_state = sentence[i-1]
            prev2_word, prev2_state = sentence[i-2]
            transition_count[prev2_state][prev_state][state] += 1

        if word not in word_count[state]:
            word_count[state][word] = 0
        word_count[state][word] += 1

transition_prob = {}
emission_prob = {}

# Calculate transition probabilities with Laplace smoothing
for state1 in transition_count:
    transition_prob[state1] = {}
    for state2 in transition_count[state1]:
        total_transitions = sum(transition_count[state1][state2].values())
        total_transitions_with_smoothing = total_transitions + smoothing_factor * len(states)
        transition_prob[state1][state2] = {next_state: (count + smoothing_factor) / total_transitions_with_smoothing for next_state, count in transition_count[state1][state2].items()}

# Calculate emission probabilities with Laplace smoothing
for state in word_count:
    total_emissions = state_count[state]
    total_emissions_with_smoothing = total_emissions + smoothing_factor * len(word_count[state])
    emission_prob[state] = {word: (count + smoothing_factor) / total_emissions_with_smoothing for word, count in word_count[state].items()}

# Print transition probabilities
print("Transition probabilities (Trigram with Laplace smoothing):")
print(transition_prob)
print()

# Print emission probabilities
print("Emission probabilities (Trigram with Laplace smoothing):")
print(emission_prob)


# In[11]:


'''
Perform 5-fold cross-validation on the Training datasets and report both average
& individual fold results (Accuracy, Precision, Recall and F-Score).
'''

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
fscores = []

for train_indices, test_indices in kfold.split(dataset):
    train_set = [dataset[i] for i in train_indices]
    test_set = [dataset[i] for i in test_indices]

    # Evaluate the model on the test set
    y_true = []
    y_pred = []

    for sentence in test_set:
        tokens = [token for token, _ in sentence]
        true_tags = [tag for _, tag in sentence]

        # Make predictions using the trained model
        predicted_tags = viterbi(tokens, states, transition_prob, emission_prob, start_prob)

        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)

    # Calculate evaluation metrics for the fold
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    fscore = f1_score(y_true, y_pred, average='weighted')

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)

    # Print the evaluation metrics for the fold
    print(f"Fold results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F-Score: {fscore}")

# Calculate average evaluation metrics across all folds
avg_accuracy = sum(accuracies) / len(accuracies)
avg_precision = sum(precisions) / len(precisions)
avg_recall = sum(recalls) / len(recalls)
avg_fscore = sum(fscores) / len(fscores)

# Print the average evaluation metrics
print("Average results across all folds:")
print(f"Accuracy: {avg_accuracy}, Precision: {avg_precision}, Recall: {avg_recall}, F-Score: {avg_fscore}")


# In[ ]:




