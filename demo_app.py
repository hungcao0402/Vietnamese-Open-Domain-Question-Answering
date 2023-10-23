import streamlit as st

import requests


def predict(input_text):
    url = "http://localhost:8088/predict/"  # Update with your server's URL

    # Create a dictionary with the input text
    data = {"input_text": input_text}

    # Send the POST request
    response = requests.post(url, json=data)

    # Check the response
    if response.status_code == 200:
        result = response.json()
        output_text = result["output_text"]
        return output_text, result['top5']
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        return ''

# Streamlit app title and description
st.title("Open Domain Question Answering")
st.write("Ask a question, and I will do my best to answer it!")

# Input for user question
user_question = st.text_input("Ask a question:", value='chủ tịch nước đầu tiên của Việt Nam là ai?')


# Button to trigger question answering
if st.button("Answer"):
    answer, top5 = predict(user_question)
    st.write(f"**Answer:** {answer}")
    st.write(f'**Top 5 context:**')
    for line in top5:
        st.write("<Passage: {},\n score:{}>".format(line['context'], line['score']))
