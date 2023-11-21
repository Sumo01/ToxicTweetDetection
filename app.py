import streamlit as st
from Utils import predictor, retrainer


def main():
    st.title("Tweet Classification App")
    tweet_input = st.text_area("Enter your tweet:")
    classification_method = st.radio("Select classification method:", ['A', 'B', 'C'])

    if st.button("Classify"):
        # Make predictions based on the selected method
        prediction = predictor.predict(tweet_input, classification_method)

        # Display the result
        st.success(f"The predicted class using Method {classification_method} is: {prediction}")

if __name__ == "__main__":
    main()