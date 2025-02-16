from rest_framework.views import APIView
from rest_framework import viewsets
from django.template.loader import render_to_string
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User
 
from rest_framework.decorators import action
from django.shortcuts import render
 
from django.views.decorators.csrf import csrf_exempt
import secrets
from django.conf import settings
import re
import json, os, zipfile
import time
import logging
import csv
import whisper, torch, traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from rest_framework.parsers import MultiPartParser, FormParser
import os
import warnings

from pydub import AudioSegment
from pydub.silence import split_on_silence

# Ignore all warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
# Set the CUDA_LAUNCH_BLOCKING environment variable
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


info_logger = logging.getLogger('api_info')
error_logger = logging.getLogger('api_error')
 

# print(torch.cuda.is_available())  # Should return True if GPU is available
# print(torch.cuda.device_count())  # Number of GPUs available
# print(torch.cuda.get_device_name(0))

# Loading Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model(str(settings.BASE_DIR)+"/model/large-v2.pt",  device=device)
# # Define the model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
print(f"Device: {device}")
print(f"Model Loaded.")
 
class IVR_APIS(APIView):
    parser_classes = [MultiPartParser, FormParser]
    def get(self, request):
        try:
            # return render(request, 'index.html')
            return Response({"status": "Failure", "message": "Get method Not ALLOWED"}, status=400)
        except Exception as e:
            print("Exception in IVR_APIS get", e)
            
   
    

    def is_valid_account_number(self, account_number):
        account_number_pattern = (
        r"^(?:4[0-9]{12}(?:[0-9]{3})?"            # Visa
        r"|5[1-5][0-9]{14}"                      # MasterCard (Old Range)
        r"|2(?:2[2-9][0-9]{2}|[3-6][0-9]{3}|7[0-1][0-9]{2}|720)[0-9]{12}"  # MasterCard (New Range)
        r"|3[47][0-9]{13}"                       # American Express
        r"|3(?:0[0-5]|[68][0-9])[0-9]{11}"      # Diners Club
        r"|6(?:011|5[0-9]{2}|22(?:12[6-9]|1[3-9][0-9]|[2-8][0-9]{2}|9[0-1][0-9]|92[0-5]))[0-9]{12}"  # Discover
        r"|(?:2131|1800|35\d{3})\d{11})$"       # JCB
         )
        
        return bool(re.match(account_number_pattern, account_number))
    

    def extract_data(self, user_input, bit):
        pt_acc_prompt = f"""
            Extract and present the patient account number from the user's input in the specified format:
            "xxxxxxxxxxxxxxxx"
            Please extract the patient account number and present it in the specified format,
            making sure to replace the "x" with the actual numbers you extracted from the user's input.
            Convert all words to numerical form accurately:
            Text: "{user_input}"
 
            Output only the extracted and formatted number. Do not include any additional questions or answers.
        """
        
        cc_prompt = f"""
            Extract and present the credit card number from the user's input in the specified format:
            "xxxxxxxxxxxxxxxx"
            Please extract the credit card number and present it in the specified format,
            making sure to replace the "x" with the actual numbers you extracted from the user's input.
            Convert all words to numerical form accurately:
            Text: "{user_input}"
 
            Output only the extracted and formatted number. Do not include any additional questions or answers.
        """
 
        date_prompt = f"""
            Extract the expiry date from the user's input and present it in the "MM/YY" format:
            - "MM" is the two-digit numerical representation of the month (e.g., "January" becomes "01", "March" becomes "03").
            - "YY" is the last two digits of the year (e.g., "2025" becomes "25", "2027" becomes "27").
            
            Important Guidelines:
            1. If the input contains a month and a two-digit number (e.g., "November 29"), treat the two-digit number as the year.
            2. If the input contains a month and a four-digit number (e.g., "November 2029"), convert the year to its last two digits.
            3. If the input contains only numbers (e.g., "7 25"), interpret them as "MM YY".
            4. Ignore any text that is not part of the date.

            Text: "{user_input}"

            Output only the extracted and formatted date in "MM/YY" format. Do not include explanations, additional text, or questions.
        """

 
        cvc_prompt = f"""
            Extract and present the Card Verification Code (CVC) / Card Verification Value (CVV) from the user's input in the specified format:
            "XXX"
            Please extract the CVC/CVV and present it in the specified format,
            making sure to replace the "XXX" with the actual numbers you extracted from the user's input.
            Convert all words to numerical form accurately:
            Text: "{user_input}"
 
            Output only the extracted and formatted number. Do not include any additional questions or answers.
        # """
 
        if bit==0:
            prompt = pt_acc_prompt
        elif bit==1:
            prompt = cc_prompt
        elif bit==2:
            prompt = date_prompt
        elif bit==3:
            prompt = cvc_prompt
 
        # Construct the full prompt
        full_prompt = f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nUser: {prompt}\nAssistant:"
 
        # Tokenize the input
        model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        print(device)
 
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
        if bit == 1 and not self.is_valid_account_number(response):
            return {"status": "Failed", "message": "Account number is not valid"}
        return {"status": "Success", "message": response}
    
    def post(self, request):
        try:
            # Specify the local audio file path
            audio_file_path = str(settings.BASE_DIR) + "/uploaded_files/Recording.mp3"

            if not os.path.exists(audio_file_path):
                print(audio_file_path)
                return Response({"status": "Failure", "message": "Audio file does not exist at path"}, status=400)

           
            generator = self.process_audio(audio_file_path)

            account_number = next(generator)  
            print("Extracted Account Number:", account_number)

            credit_card_number = next(generator) 
            print("Extracted Credit Card Number:", credit_card_number)
            csv_file_path = str(settings.BASE_DIR) + "/IVR_Record_All.csv"

           
            columns = ["Audio File Location", "Account Number", "Credit Card Number"]
            file_exists = os.path.isfile(csv_file_path)

            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)

                if not file_exists:
                    writer.writerow(columns)

                writer.writerow([audio_file_path, account_number, credit_card_number])


            return Response({
                "status": "Success",
                "account_number": account_number,
                "credit_card_number": credit_card_number
            }, status=200)

        except StopIteration:
            return Response({"status": "Failure", "message": "Audio transcription incomplete"}, status=400)
        except Exception as e:
            traceback.print_exc()
            return Response({"status": "Failure", "message": str(e)}, status=500)

    def process_audio(self, audio_file_path):
        """
        Generator function to process audio in chunks based on silence detection.
        It transcribes the audio, extracts account number first, then credit card number, and yields each.
        """
        audio = AudioSegment.from_file(audio_file_path)


        chunks = split_on_silence(
            audio,
            min_silence_len=1000,  
            silence_thresh=-40  
        )

        if len(chunks) < 2:
            raise ValueError("Insufficient data in transcription for account number and credit card number.")

        first_chunk_path = str(settings.BASE_DIR) + "/temp_first_chunk.wav"
        chunks[0].export(first_chunk_path, format="wav")
        account_transcription = whisper_model.transcribe(first_chunk_path, fp16=False, language='english')['text']
        account_number = self.extract_account_number(account_transcription)
        yield account_number  


        remaining_audio = AudioSegment.silent(duration=0)
        for chunk in chunks[1:]:
            remaining_audio += chunk
        remaining_chunk_path = str(settings.BASE_DIR) + "/temp_remaining_chunks.wav"
        remaining_audio.export(remaining_chunk_path, format="wav")
        credit_card_transcription = whisper_model.transcribe(remaining_chunk_path, fp16=False, language='english')['text']
        credit_card_number = self.extract_credit_card_number(credit_card_transcription)
        yield credit_card_number  # Yield the credit card number

    def extract_account_number(self, text):
        data_type = 0
        acc_number = self.extract_data(text, data_type)
        return acc_number

    def extract_credit_card_number(self, text):

        data_type = 1
        CC_number = self.extract_data(text, data_type)
        return CC_number