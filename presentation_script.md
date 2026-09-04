# Video Presentation Script: Customer Satisfaction (CSAT) Prediction

**Total Duration:** ~15-20 minutes
**Instructions:** Speak clearly, share your screen to show the notebook and Streamlit app, and follow this script to cover all evaluation criteria.

---

## 1. Introduction (1.5 min)
*Screen: Title slide or the top of the Google Colab Notebook.*

"Hello everyone, my name is [Your Name], and today I am presenting my deep learning project focused on predicting Customer Satisfaction, or CSAT scores, in the e-commerce sector. 
The problem domain is e-commerce customer support, where understanding satisfaction is critical for retention.
For this project, I built an Artificial Neural Network (ANN) using Python and deep learning libraries like TensorFlow and Keras. 
The objective of this model is to accurately predict the CSAT score (ranging from 1 to 5) based on various interaction features like handling time, channel used, and categorical data. 
Later in this presentation, I'll also demonstrate how I deployed this model locally using Streamlit."

---

## 2. Problem Understanding (2 min)
*Screen: Present a slide or just speak to the camera.*

"Predicting CSAT is incredibly useful for e-commerce because customer satisfaction directly impacts business outcomes like customer loyalty, repeat purchases, and word-of-mouth marketing. 
Traditionally, companies rely on manual surveys. However, surveys have notoriously low response rates and are reactive rather than proactive. 
By building a predictive model, we can estimate the satisfaction of *every* customer interaction in real-time, even if they don't fill out a survey. This supports service quality improvement by allowing support teams to instantly identify and resolve poor interactions, ultimately saving the customer relationship."

---

## 3. Data Understanding & Preparation (2 min)
*Screen: Show the Data Loading and Cleaning sections of the Colab notebook.*

"Moving on to the data. The dataset contains features like `channel_name`, `category`, `connected_handling_time`, `Agent Shift`, and timestamps for issue reporting and resolution. 
During the data cleaning process, I identified missing values and dropped irrelevant columns like `Unique id` and `Order_id`.
For feature engineering, I extracted a new feature called `response_time_mins` by subtracting the `Issue reported at` timestamp from `issue_responded`. I also created a binary feature `has_remarks` based on whether the customer left text feedback, keeping the model tabular and efficient.
Finally, I applied categorical one-hot encoding for text fields and Standard Scaling for continuous numerical fields like `Item_price` and `handling_time` to prepare the data for the neural network."

---

## 4. ANN Architecture & Training Process (2.5 min)
*Screen: Show the Model Architecture and Training sections in the Colab notebook.*

"I chose an Artificial Neural Network (ANN) because it excels at capturing complex, non-linear relationships across a mix of numerical and categorical features. 
The architecture consists of an input layer, followed by three hidden Dense layers with 128, 64, and 32 neurons respectively, using ReLU activation functions. 
Because the CSAT score is an integer from 1 to 5, I framed this as a multi-class classification problem. Thus, the output layer has 5 neurons with a Softmax activation function.
To prevent overfitting, I introduced Dropout layers. I also implemented training optimization callbacks, specifically Early Stopping and Learning Rate reduction on plateau. I trained the model on an 85/15 train-test split, ensuring generalizability."

---

## 5. Model Evaluation & Prediction Quality (2 min)
*Screen: Show the Model Evaluation section, classification report, and confusion matrix.*

"To evaluate performance, I used categorical cross-entropy loss and accuracy. 
Looking at the classification report and confusion matrix, we can see how the model differentiates between the 5 satisfaction levels. 
The model reliably identifies highly satisfied customers (CSAT 5) and highly dissatisfied ones (CSAT 1). 
Common error patterns usually occur between adjacent scores, like confusing a 4 with a 5, which is expected since human ratings can be subjective. Overall, the predictions reveal that rapid response times and specific channels heavily influence positive satisfaction trends."

---

## 6. Business Insights & Local Deployment (2 min)
*Screen: Switch screen sharing to the Streamlit App running locally.*

"Now, let's look at how this translates to business value. I deployed this model locally using Streamlit. 
*(Demonstrate inputting some dummy data into the app and clicking predict).*
As you can see, customer support teams can use this dashboard to instantly gauge the outcome of an interaction. 
Service improvement opportunities revealed by the model show that reducing handling time in specific sub-categories dramatically improves CSAT. Operationally, a business could use this to flag 'predicted low CSAT' interactions for immediate managerial review."

---

## 7. Challenges, Optimization & Improvements (1.5 min)
*Screen: Keep the Streamlit app on screen or switch back to the notebook.*

"Throughout this project, I faced several challenges. 
One major challenge was the high cardinality of categorical features, like agent names, which led to overfitting. I optimized this by focusing on higher-level features like `Agent Shift` and `Tenure Bucket`.
Another challenge was setting up the local deployment pipeline to seamlessly load the preprocessor and the Keras model.
In the future, I'd like to improve the model by using text embeddings like Word2Vec or BERT to actually analyze the sentiment of the `Customer Remarks`, rather than just checking if a remark exists."

---

## 8. Learnings & Practical Value (1.5 min)
*Screen: Concluding slide or camera view.*

"To conclude, applying deep learning to a tangible business problem taught me the immense value of end-to-end machine learning pipelines—from raw data to a deployed application.
The most important insight I found was the direct correlation between automated feature extraction (like response time) and the CSAT score. 
The proactive value of this system is massive: instead of waiting weeks for survey results, businesses can act in minutes. 
Thank you for watching my presentation."

---

### Potential Interview QA Prep:
- **Why did you choose ANN for this problem?** Because ANNs can effectively model non-linear interactions between diverse tabular features (mixed categorical and numerical) better than simple linear models.
- **Which features were most useful?** Typically response time, handling time, and issue category.
- **How would you reduce overfitting?** Using Dropout layers (as implemented), reducing model complexity, or using L2 regularization.
- **How would you improve the system for real-time prediction?** Deploy the model via an API (like FastAPI or Flask) hosted on a cloud platform (AWS/GCP), integrated directly into the customer service CRM platform.
