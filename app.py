import streamlit as st
from Utils import predictor, retrainer


def main():
    st.title("Is this tweet toxic?")
    tweet_input = st.text_area("Enter a tweet to check if its offensive or not:")
    classification_method = st.radio("Select classification method:", ['A', 'B', 'C'])

    if st.button("Classify"):
        # Make predictions based on the selected method
        prediction = predictor.predict(tweet_input, classification_method)

        # Display the result
        st.success(f"The predicted class using Method {classification_method} is: {prediction}")
        
    st.write("A - Offensive or Not Offensive")
    st.write("B - Offensive and Directed or Offensive and Not Directed")
    st.write("C - Offensive and Directed to an Individual, Offensive and Directed to a Group OR Other")

if __name__ == "__main__":
    main()