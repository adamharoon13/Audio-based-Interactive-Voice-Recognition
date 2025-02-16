from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, re
 
# # Define the model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
 
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"  # Automatically map to GPU if available
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
import re
 
# Updated pattern for credit card validation
pattern = (
    r"^(?:4[0-9]{12}(?:[0-9]{3})?"            # Visa
    r"|5[1-5][0-9]{14}"                      # MasterCard (Old Range)
    r"|2(?:2[2-9][0-9]{2}|[3-6][0-9]{3}|7[0-1][0-9]{2}|720)[0-9]{12}"  # MasterCard (New Range)
    r"|3[47][0-9]{13}"                       # American Express
    r"|3(?:0[0-5]|[68][0-9])[0-9]{11}"      # Diners Club
    r"|6(?:011|5[0-9]{2}|22(?:12[6-9]|1[3-9][0-9]|[2-8][0-9]{2}|9[0-1][0-9]|92[0-5]))[0-9]{12}"  # Discover
    r"|(?:2131|1800|35\d{3})\d{11})$"       # JCB
)
 
# Example function to validate a card number
def is_valid_credit_card(card_number):
    return bool(re.match(pattern, card_number))
 
def extract_cc(user_input):
 
    # prompt = f"""
    #     Extract and present the patient account number from the user's input in the specified format:
    #     "xxxxxxxxxxxxxxxx"
    #     Please extract the patient account number and present it in the specified format,
    #     making sure to replace the "x" with the actual numbers you extracted from the user's input.
    #     Convert all words to numerical form accurately:
    #     Text: "{user_input}"
 
    #     Output only the extracted and formatted number. Do not include any additional questions or answers.
    # """
    
    prompt = f"""
        Extract and present the credit card number from the user's input in the specified format:
        "xxxxxxxxxxxxxxxx"
        Please extract the credit card number and present it in the specified format,
        making sure to replace the "x" with the actual numbers you extracted from the user's input.
        Convert all words to numerical form accurately:
        Text: "{user_input}"
 
        Output only the extracted and formatted number. Do not include any additional questions or answers.
    """
 
    # prompt = f"""
    #     Extract and present the Date from the user's input in the specified format:
    #     "MM/YY"
    #     Please extract the date and present it in the specified format,
    #     making sure to replace the "MM/YY" with the actual month and year you extracted from the user's input.
    #     Convert all words to numerical form accurately:
    #     Text: "{user_input}"
 
    #     Output only the extracted and formatted Date. Do not include any additional questions or answers.
    # """
 
    # prompt = f"""
    #     Extract and present the Card Verification Code (CVC) / Card Verification Value (CVV) from the user's input in the specified format:
    #     "XXX"
    #     Please extract the CVC/CVV and present it in the specified format,
    #     making sure to replace the "XXX" with the actual numbers you extracted from the user's input.
    #     Convert all words to numerical form accurately:
    #     Text: "{user_input}"
 
    #     Output only the extracted and formatted number. Do not include any additional questions or answers.
    # """
 
    # Construct the full prompt
    full_prompt = f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nUser: {prompt}\nAssistant:"
 
    # Tokenize the input
    model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
 
    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=150,  
        temperature=0.1,     
        top_p=0.95,          
        do_sample=False      
    )
 
    # Decode and clean the generated text
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response  = response.split('Assistant:')[1].strip()
    print(f"response: {response}")
    is_valid = is_valid_credit_card(response)
    print(response, " : ", is_valid)
 
# Define your user input and construct the prompt
# user_input = "It’s 4526, three-six-nine-eight, five-six-nine-five, six-three-six-nine, just as I have it here."
cc_list = [
    "4111111111111111",
    "5500000000000004",
    "340000000000009",
    "30000000000004",
    "6011000000000004",
    "2014000000000000",
    "3088000000000009",
    "3576000000000000",
    "6331100000000000",
    "3530111333300000",
    "5105105105105100",
    "6011514433545378",
    "3530111333300001",
    "6771555500000000",
    "5403988389988005",
    "6034999900000002",
    "3566002020360505",
    "3842022001000100",
    "5226394748366741",
    "4716994117321100"
]


 
for user_input in cc_list:
    extract_cc(user_input)