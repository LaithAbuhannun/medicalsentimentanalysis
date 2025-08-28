# Medical Sentiment Analysis with DistilBERT  

## Overview  
This project uses **DistilBERT** to analyze sentiment in medical and mental health conversations. It also detects self-harm related language to help flag high-risk cases.  

---

## Problem  
- People hide distress in vague or sarcastic language (*“I’m fine”* can mean the opposite).  
- Traditional sentiment tools and even professionals can miss these cues.  

---

## Solution  
The system:  
1. Classifies text as **Positive, Neutral, or Negative**  
2. Detects **self-harm keywords**  
3. Flags risky responses for early intervention  

---

## How It Works  
- Model: DistilBERT (fine-tuned)  
- Dataset: 1,500 labeled sentences  
- Training: 20 epochs, batch size 16  
- Extra: Rule-based keyword detection (e.g., “I want to disappear”)  
- Demo: Voice recorder (HTML + JS)  

---

## Results  
- Correctly identified negative sentences (*“I had a pretty bad day”*)  
- Detected self-harm intent (*“I want to commit suicide”*)  
- System is scalable and works in real time  




