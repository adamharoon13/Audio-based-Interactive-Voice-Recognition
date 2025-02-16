import ollama

# User-provided transcription input
user_input = "card number is 5 9 8 4 6 5 2 1 1 3 0 8 1 1 2 9 5 oh no no no no last one is wrong that 1 1 2 1 this is correct"

response = ollama.chat(
    model="llama3.1",
    messages=[
        {
            "role": "system", 
            "content": (
                "You are an expert at extracting and formatting numerical sequences from text. "
                "Your task is to identify and format 16-digit sequences from the input text, "
                "even if the digits are written as words or split into groups. "
                "If the sequence is written in words (e.g., 'four five two six'), convert it to digits (e.g., '4526'). "
                "Only return the digits in the required format (e.g., '1234 4565 1213 4321') or 'Not found' if no sequence is found. "
                "Do not provide any additional text or explanations. Precision is critical in this task."
            ),
        },
        {
            "role": "user", 
            "content": f"Input text: \"{user_input}\""
        },
    ],
)
print(response["message"]["content"])





# import ollama

# # User-provided transcription input
# user_input = "The expiry is zero seven twenty-five."

# response = ollama.chat(
#     model="llama3.1",
#     messages=[
#         {
#             "role": "system", 
#             "content": (
#                 "You are an expert at extracting and formatting expiry dates from text. "
#                 "Your task is to identify and format expiry dates from the input text. "
#                 "The expiry date can be in various formats such as written out in words (e.g., 'zero seven twenty-five'), "
#                 "numeric format (e.g., '07/25'), or mixed formats (e.g., 'July 2025'). "
#                 "Convert all expiry dates to the standard MM/YY format (e.g., '07/25'). "
#                 "If no expiry date is found, return ''. "
#                 "Do not provide any additional text or explanations. Precision is critical in this task."
#             ),
#         },
#         {
#             "role": "user", 
#             "content": f"Input text: \"{user_input}\""
#         },
#     ],
# )
# print(response["message"]["content"])




# import ollama

# # User-provided transcription input
# user_input = "security code is 321"

# response = ollama.chat(
#     model="llama3.1",
#     messages=[
#         {
#             "role": "system", 
#             "content": (
#                 "You are an expert at extracting and formatting 3-digit numerical codes from text. "
#                 "Your task is to identify and extract 3-digit codes from the input text. "
#                 "The code can be in various formats: written as words (e.g., 'two nine six'), numeric format (e.g., '296'), "
#                 "or a mixed format (e.g., 'The code is two nine six'). "
#                 "Only return the 3-digit code in numeric form (e.g., '296'). "
#                 "If no 3-digit code is found, return an empty string (''). "
#                 "Do not provide any additional text or explanations. Precision is critical in this task."
#             ),
#         },
#         {
#             "role": "user", 
#             "content": f"Input text: \"{user_input}\""
#         },
#     ],
# )
# print(response["message"]["content"])