import streamlit as st
from Utils import predictor, retrainer


def main():
    st.title("Tweet Classification App")

    # User input for the tweet
    tweet_input = st.text_area("Enter your tweet:")
    if st.button("Classify"):
        # Make predictions
        prediction = predictor.predict(tweet_input)
        if prediction==0: pclass="Not offensive" 
        else: pclass="Offensive"
        # Display the result
        st.success(f"The predicted class is: {pclass}")

if __name__ == "__main__":
    main()